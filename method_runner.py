import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import generator
from random import random, seed
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


"""
This file contains essential part of the code.

It is highly advised to leave this file be as it is.
"""


# Number of processes for multiprocessing 
processes = 2

def search_learning_rate(metric, data, search_decay=False):
    best_learning_rate = list(data.keys())[0]
    if not search_decay:
        best_val = data[best_learning_rate][metric]
        for learning_rate in data.keys():
            if data[learning_rate][metric] <= best_val:
                best_val = data[learning_rate][metric]
                best_learning_rate = learning_rate
        return (metric, best_learning_rate)
    else:
        best_decay = list(data[best_learning_rate].keys())[0]
        best_val = data[best_learning_rate][best_decay][metric]

        for learning_rate in data.keys():
            for decay in data[learning_rate].keys():
                if data[learning_rate][decay][metric] < best_val:
                    best_val = data[learning_rate][decay][metric]
                    best_decay = decay
                    best_learning_rate = learning_rate
        return (metric, (best_learning_rate, best_decay))

class MethodRunner:
    def __init__(self, seed_counter, iterations, m, radius, dec, step, train_losses, test_losses, save=''):
        self.seed_counter = seed_counter
        self.iterations = iterations
        self.m = m
        self.radius = radius
        self.dec = dec
        self.step = step
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.save = save
        self.x = [i for i in range(self.iterations)]
        if save:
            with open(save + 'test_results.csv', 'w') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                thewriter.writeheader()
    
    @staticmethod
    def show_plot():
        plt.show()

    def runGradientDescent(self, learning_rates):

        ## Calculate best params for each learinig rate
        def train(learning_rate):
            tf.reset_default_graph()
            temp = np.empty(0, np.float32)
            for loss in self.train_losses.keys():
                for dim in self.train_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.train_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.train_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])
                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate, global_step, self.step, self.dec, name='train_gradient')).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                temp = np.append(temp, f_curr)
            return learning_rate, {'best': np.min(temp).item(), 'worst': np.max(temp).item(), 'mean': np.mean(temp).item(), "median": np.median(temp).item()}

        ## Parallel training
        with Pool(processes) as p:
            train_results = p.map(train, learning_rates)
            train_results = {x: y for (x, y) in train_results}

        # print('Gradient Descent results:')
        # print(json.dumps(train_results, indent=4), '\n')

        with open(self.save + 'gd_test_results.json', 'w') as file:
            json.dump(train_results, file)

        print('Gradient training done')
        
        search_learning_rate_partial = partial(search_learning_rate, data=train_results)
        with Pool(processes) as p:
            metrics = ['best', 'worst', 'mean', 'median']
            best_params = p.map(search_learning_rate_partial, metrics)
        best_params = {x: y for (x, y) in best_params}
        
        # print('Best gradient descent learning rates:', " ".join([str(i) for i in best_params]), '\n')

        # Test function
        def test(input_data):
            learning_rate = input_data[0]
            metric = input_data[1]
            metric_name = input_data[2]

            tf.reset_default_graph()
            results = np.empty(shape=(0, self.iterations), dtype=np.float32)

            for loss in self.test_losses.keys():
                for dim in self.test_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.test_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.test_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate.astype(np.float32), global_step, self.step, self.dec, name='test_gradient')).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                temp = np.empty(self.iterations)
                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                    temp[i] = f_curr

                                results = np.append(results, np.array([temp]), axis=0)
            return (metric_name, metric(results, axis=0))

        with Pool(processes) as p:
            pairs = [[best_params['best'], np.min, 'best'], [best_params['worst'], np.max, 'worst'], [best_params['mean'], np.mean, 'mean'], [best_params['median'], np.median, 'median']]
            test_results = p.map(test, pairs)
        test_results = {x: y for (x, y) in test_results}

        fig = plt.figure()
        plt.plot(self.x, test_results['best'], 'r', alpha=0.6, label='Best, LR:' + str(round(best_params['best'], 5)))
        plt.plot(self.x, test_results['worst'], 'm-o', alpha=0.3, label='Worst, LR:' + str(round(best_params['worst'], 5)))
        plt.plot(self.x, test_results['mean'], 'g-*', alpha=0.3, label='Mean, LR:' + str(round(best_params['mean'], 5)))
        plt.plot(self.x, test_results['median'], 'b--', alpha=0.6, label='Median, LR:' + str(round(best_params['median'], 5)))
        plt.title('Gradient descent')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'gd_train.png', dpi=300)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'best', 'iteration': i, 'value': test_results['best'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'worst', 'iteration': i, 'value': test_results['worst'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'mean', 'iteration': i, 'value': test_results['mean'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'median', 'iteration': i, 'value': test_results['median'][i]})

        print('Gradient testing done\n')

    def runAdam(self, learning_rates):

        ## Calculate best params for each learinig rate
        def train(learning_rate):
            tf.reset_default_graph()
            temp = np.empty(0, np.float32)
            for loss in self.train_losses.keys():
                for dim in self.train_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.train_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.train_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])
                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate, global_step, self.step, self.dec, name='train_gradient')).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                temp = np.append(temp, f_curr)
            return learning_rate, {'best': np.min(temp).item(), 'worst': np.max(temp).item(), 'mean': np.mean(temp).item(), "median": np.median(temp).item()}

        ## Parallel training
        with Pool(processes) as p:
            train_results = p.map(train, learning_rates)
            train_results = {x: y for (x, y) in train_results}

        # print('Adam results:')
        # print(json.dumps(train_results, indent=4), '\n')

        with open(self.save + 'adam_test_results.json', 'w') as file:
            json.dump(train_results, file)

        print('Adam training done')

        search_learning_rate_partial = partial(search_learning_rate, data=train_results)
        with Pool(processes) as p:
            metrics = ['best', 'worst', 'mean', 'median']
            best_params = p.map(search_learning_rate_partial, metrics)
        best_params = {x: y for (x, y) in best_params}

        # print('Best adam learning rates:', " ".join([str(i) for i in best_params]), '\n')

        # Test function
        def test(input_data):
            learning_rate = input_data[0]
            metric = input_data[1] 
            metric_name = input_data[2]

            tf.reset_default_graph()
            results = np.empty(shape=(0, self.iterations), dtype=np.float32)

            for loss in self.test_losses.keys():
                for dim in self.test_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.test_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.test_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate.astype(np.float32), global_step, self.step, self.dec, name='test_gradient')).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                temp = np.empty(self.iterations)
                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                    temp[i] = f_curr

                                results = np.append(results, np.array([temp]), axis=0)
            return (metric_name, metric(results, axis=0))

        with Pool(processes) as p:
            pairs = [[best_params['best'], np.min, 'best'], [best_params['worst'], np.max, 'worst'], [best_params['mean'], np.mean, 'mean'], [best_params['median'], np.median, 'median']]
            test_results = p.map(test, pairs)
        test_results = {x: y for (x, y) in test_results}

        fig = plt.figure()
        plt.plot(self.x, test_results['best'], 'r', alpha=0.6, label='Best, LR:' + str(round(best_params['best'], 5)))
        plt.plot(self.x, test_results['worst'], 'm-o', alpha=0.3, label='Worst, LR:' + str(round(best_params['worst'], 5)))
        plt.plot(self.x, test_results['mean'], 'g-*', alpha=0.3, label='Mean, LR:' + str(round(best_params['mean'], 5)))
        plt.plot(self.x, test_results['median'], 'b--', alpha=0.6, label='Median, LR:' + str(round(best_params['median'], 5)))
        plt.title('Adam')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'adam_train.png', dpi=300)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'best', 'iteration': i, 'value': test_results['best'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'worst', 'iteration': i, 'value': test_results['worst'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'mean', 'iteration': i, 'value': test_results['mean'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'median', 'iteration': i, 'value': test_results['median'][i]})

        print('Adam testing done\n')

    def runMomentum(self, learning_rates):
        ## Calculate best params for each learinig rate
        def train(learning_rate):
            tf.reset_default_graph()
            temp = np.empty(0, np.float32)
            for loss in self.train_losses.keys():
                for dim in self.train_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.train_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.train_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])
                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.MomentumOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate, global_step, self.step, self.dec, name='train_gradient'), momentum=0.999, use_nesterov=True).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.999, use_nesterov=True).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                temp = np.append(temp, f_curr)
            return learning_rate, {'best': np.min(temp).item(), 'worst': np.max(temp).item(), 'mean': np.mean(temp).item(), "median": np.median(temp).item()}

        ## Parallel training
        with Pool(processes) as p:
            train_results = p.map(train, learning_rates)
            train_results = {x: y for (x, y) in train_results}

        # print('Momentum results:')
        # print(json.dumps(train_results, indent=4), '\n')

        with open(self.save + 'momentum_test_results.json', 'w') as file:
            json.dump(train_results, file)

        print('Momentum training done')

        search_learning_rate_partial = partial(search_learning_rate, data=train_results)
        with Pool(processes) as p:
            metrics = ['best', 'worst', 'mean', 'median']
            best_params = p.map(search_learning_rate_partial, metrics)
        best_params = {x: y for (x, y) in best_params}

        # print('Best momentum learning rates:', " ".join([str(i) for i in best_params]), '\n')

        # Test function
        def test(input_data):
            learning_rate = input_data[0]
            metric = input_data[1]
            metric_name = input_data[2] 

            tf.reset_default_graph()
            results = np.empty(shape=(0, self.iterations), dtype=np.float32)

            for loss in self.test_losses.keys():
                for dim in self.test_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.test_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.test_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.MomentumOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate.astype(np.float32), global_step, self.step, self.dec, name='test_gradient'), momentum=0.999, use_nesterov=True).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.999, use_nesterov=True).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                temp = np.empty(self.iterations)
                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                    temp[i] = f_curr

                                results = np.append(results, np.array([temp]), axis=0)
            return (metric_name, metric(results, axis=0))

        with Pool(processes) as p:
            pairs = [[best_params['best'], np.min, 'best'], [best_params['worst'], np.max, 'worst'], [best_params['mean'], np.mean, 'mean'], [best_params['median'], np.median, 'median']]
            test_results = p.map(test, pairs)
        test_results = {x: y for (x, y) in test_results}

        fig = plt.figure()
        plt.plot(self.x, test_results['best'], 'r', alpha=0.6, label='Best, LR:' + str(round(best_params['best'], 5)))
        plt.plot(self.x, test_results['worst'], 'm-o', alpha=0.3, label='Worst, LR:' + str(round(best_params['worst'], 5)))
        plt.plot(self.x, test_results['mean'], 'g-*', alpha=0.3, label='Mean, LR:' + str(round(best_params['mean'], 5)))
        plt.plot(self.x, test_results['median'], 'b--', alpha=0.6, label='Median, LR:' + str(round(best_params['median'], 5)))
        plt.title("Momentum")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'momentum_train.png', dpi=300)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'best', 'iteration': i, 'value': test_results['best'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'worst', 'iteration': i, 'value': test_results['worst'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'mean', 'iteration': i, 'value': test_results['mean'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'median', 'iteration': i, 'value': test_results['median'][i]})

        print('Momentum testing done\n')

    def runAdagrad(self, learning_rates):
        ## Calculate best params for each learinig rate
        def train(learning_rate):
            tf.reset_default_graph()
            temp = np.empty(0, np.float32)
            for loss in self.train_losses.keys():
                for dim in self.train_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.train_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.train_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])
                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.AdagradOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate, global_step, self.step, self.dec, name='train_gradient')).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                temp = np.append(temp, f_curr)
            return learning_rate, {'best': np.min(temp).item(), 'worst': np.max(temp).item(), 'mean': np.mean(temp).item(), "median": np.median(temp).item()}

        ## Parallel training
        with Pool(processes) as p:
            train_results = p.map(train, learning_rates)
            train_results = {x: y for (x, y) in train_results}

        # print('Adagrad results:')
        # print(json.dumps(train_results, indent=4), '\n')

        with open(self.save + 'adagrad_test_results.json', 'w') as file:
            json.dump(train_results, file)

        print('Adagrad training done')

        search_learning_rate_partial = partial(search_learning_rate, data=train_results)
        with Pool(processes) as p:
            metrics = ['best', 'worst', 'mean', 'median']
            best_params = p.map(search_learning_rate_partial, metrics)
        best_params = {x: y for (x, y) in best_params}


        # print('Best adam learning rates:', " ".join([str(i) for i in best_params]), '\n')

        # Test function
        def test(input_data):
            learning_rate = input_data[0]
            metric = input_data[1] 
            metric_name = input_data[2]

            tf.reset_default_graph()
            results = np.empty(shape=(0, self.iterations), dtype=np.float32)

            for loss in self.test_losses.keys():
                for dim in self.test_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.test_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.test_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.AdagradOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate.astype(np.float32), global_step, self.step, self.dec, name='test_gradient')).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                temp = np.empty(self.iterations)
                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                    temp[i] = f_curr

                                results = np.append(results, np.array([temp]), axis=0)
            return (metric_name, metric(results, axis=0))

        with Pool(processes) as p:
            pairs = [[best_params['best'], np.min, 'best'], [best_params['worst'], np.max, 'worst'], [best_params['mean'], np.mean, 'mean'], [best_params['median'], np.median, 'median']]
            test_results = p.map(test, pairs)
        test_results = {x: y for (x, y) in test_results}

        fig = plt.figure()
        plt.plot(self.x, test_results['best'], 'r', alpha=0.6, label='Best, LR:' + str(round(best_params['best'], 5)))
        plt.plot(self.x, test_results['worst'], 'm-o', alpha=0.3, label='Worst, LR:' + str(round(best_params['worst'], 5)))
        plt.plot(self.x, test_results['mean'], 'g-*', alpha=0.3, label='Mean, LR:' + str(round(best_params['mean'], 5)))
        plt.plot(self.x, test_results['median'], 'b--', alpha=0.6, label='Median, LR:' + str(round(best_params['median'], 5)))
        plt.title("Adagrad")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'adagrad_train.png', dpi=300)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'best', 'iteration': i, 'value': test_results['best'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'worst', 'iteration': i, 'value': test_results['worst'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'mean', 'iteration': i, 'value': test_results['mean'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'median', 'iteration': i, 'value': test_results['median'][i]})

        print('Adagrad testing done\n')

    def runAdadelta(self, learning_rates, decays):
        ## Calculate best params for each learinig rate
        def train(learning_rate, decays):
            tf.reset_default_graph()
            decay_dict = {}
            for decay in decays:
                temp = np.empty(0, np.float32)
                for loss in self.train_losses.keys():
                    for dim in self.train_losses[loss][1].keys():
                        seed(self.seed_counter)
                        function_generator = generator.FunGen(dim, self.train_losses[loss][1][dim], loss)
                        functions = function_generator.generate(self.train_losses[loss][0])
                        for f in functions:
                            points = generator.generate_points(dim, self.m, self.radius, f[1])
                            with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                                global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                                if self.step > 0:
                                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate, global_step, self.step, self.dec, name='train_gradient'), rho=decay).minimize(f[0], global_step=global_step)
                                else:
                                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=decay).minimize(f[0])

                            for point in points:
                                with tf.Session() as sess:
                                    sess.run(tf.global_variables_initializer())
                                    sess.run(function_generator.x.assign(point))

                                    # Normalization
                                    starting_value = sess.run(f[0])
                                    f_temp = tf.divide(f[0], tf.constant(starting_value))

                                    for i in range(self.iterations):
                                        _, f_curr = sess.run([optimizer, f_temp])
                                    temp = np.append(temp, f_curr)
                decay_dict[decay] = {'best': np.min(temp).item(), 'worst': np.max(temp).item(), 'mean': np.mean(temp).item(), "median": np.median(temp).item()}
            return learning_rate, decay_dict

        train_partial = partial(train, decays=decays)
        ## Parallel training
        with Pool(processes) as p:
            train_results = p.map(train_partial, learning_rates)
            train_results = {x: y for (x, y) in train_results}
        # print('Adadelta results:')
        # print(json.dumps(train_results, indent=4), '\n')

        with open(self.save + 'adadelta_test_results.json', 'w') as file:
            json.dump(train_results, file)

        print('Adadelta training done')

        search_learning_rate_partial = partial(search_learning_rate, data=train_results, search_decay=True)
        with Pool(processes) as p:
            metrics = ['best', 'worst', 'mean', 'median']
            best_params = p.map(search_learning_rate_partial, metrics)
        best_params = {x: y for (x, y) in best_params}

        # print('Best adadelta learning rates and decays:', " ".join([str(i) for i in best_params]), '\n')

        # Test function
        def test(input_data):
            learning_rate = input_data[0]
            decay = input_data[1]
            metric = input_data[2] 
            metric_name = input_data[3]

            tf.reset_default_graph()
            results = np.empty(shape=(0, self.iterations), dtype=np.float32)

            for loss in self.test_losses.keys():
                for dim in self.test_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.test_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.test_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.AdadeltaOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate.astype(np.float32), global_step, self.step, self.dec, name='test_gradient'), rho=decay).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=decay).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                temp = np.empty(self.iterations)
                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                    temp[i] = f_curr

                                results = np.append(results, np.array([temp]), axis=0)
            return (metric_name, metric(results, axis=0))

        with Pool(processes) as p:
            pairs = [[best_params['best'][0], best_params['best'][1], np.min, 'best'], [best_params['worst'][0], best_params['worst'][1], np.max, 'worst'], [best_params['mean'][0], best_params['mean'][1], np.mean, 'mean'], [best_params['median'][0], best_params['median'][1], np.median, 'median']]
            test_results = p.map(test, pairs)
        test_results = {x: y for (x, y) in test_results}

        fig = plt.figure()
        plt.plot(self.x, test_results['best'], 'r', alpha=0.6, label='Best, LR: {}, DEC: {}'.format(str(round(best_params['best'][0], 5)), str(round(best_params['best'][1], 5))))
        plt.plot(self.x, test_results['worst'], 'm-o', alpha=0.3, label='Worst, LR: {}, DEC: {}'.format(str(round(best_params['worst'][0], 5)), str(round(best_params['worst'][1], 5))))
        plt.plot(self.x, test_results['mean'], 'g-*', alpha=0.3, label='Mean, LR: {}, DEC: {}'.format(str(round(best_params['mean'][0], 5)), str(round(best_params['mean'][1], 5))))
        plt.plot(self.x, test_results['median'], 'b--', alpha=0.6, label='Median, LR: {}, DEC: {}'.format(str(round(best_params['median'][0], 5)), str(round(best_params['median'][1], 5))))
        plt.title("Adadelta")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'adadelta_train.png', dpi=300)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'best', 'iteration': i, 'value': test_results['best'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'worst', 'iteration': i, 'value': test_results['worst'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'mean', 'iteration': i, 'value': test_results['mean'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'median', 'iteration': i, 'value': test_results['median'][i]})

        print('Adadelta testing done\n')

    def runRMS(self, learning_rates, decays):

        ## Calculate best params for each learinig rate
        def train(learning_rate, decays):
            tf.reset_default_graph()
            decay_dict = {}
            for decay in decays:
                temp = np.empty(0, np.float32)
                for loss in self.train_losses.keys():
                    for dim in self.train_losses[loss][1].keys():
                        seed(self.seed_counter)
                        function_generator = generator.FunGen(dim, self.train_losses[loss][1][dim], loss)
                        functions = function_generator.generate(self.train_losses[loss][0])
                        for f in functions:
                            points = generator.generate_points(dim, self.m, self.radius, f[1])
                            with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                                global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                                if self.step > 0:
                                    optimizer = tf.train.RMSPropOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate, global_step, self.step, self.dec, name='train_gradient'), decay=decay).minimize(f[0], global_step=global_step)
                                else:
                                    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(f[0])

                            for point in points:
                                with tf.Session() as sess:
                                    sess.run(tf.global_variables_initializer())
                                    sess.run(function_generator.x.assign(point))

                                    # Normalization
                                    starting_value = sess.run(f[0])
                                    f_temp = tf.divide(f[0], tf.constant(starting_value))

                                    for i in range(self.iterations):
                                        _, f_curr = sess.run([optimizer, f_temp])
                                    temp = np.append(temp, f_curr)
                decay_dict[decay] = {'best': np.min(temp).item(), 'worst': np.max(temp).item(), 'mean': np.mean(temp).item(), "median": np.median(temp).item()}
            return learning_rate, decay_dict

        train_partial = partial(train, decays=decays)
        ## Parallel training
        with Pool(processes) as p:
            train_results = p.map(train_partial, learning_rates)
            train_results = {x: y for (x, y) in train_results}

        # print('RMS results:')
        # print(json.dumps(train_results, indent=4), '\n')

        with open(self.save + 'rms_test_results.json', 'w') as file:
            json.dump(train_results, file)

        print('RMS training done')

        search_learning_rate_partial = partial(search_learning_rate, data=train_results, search_decay=True)
        with Pool(processes) as p:
            metrics = ['best', 'worst', 'mean', 'median']
            best_params = p.map(search_learning_rate_partial, metrics)
        best_params = {x: y for (x, y) in best_params}

        # print('Best RMS learning rates and decays:', " ".join([str(i) for i in best_params]), '\n')

        # Test function
        def test(input_data):
            learning_rate = input_data[0]
            decay = input_data[1]
            metric = input_data[2] 
            metric_name = input_data[3]

            tf.reset_default_graph()
            results = np.empty(shape=(0, self.iterations), dtype=np.float32)

            for loss in self.test_losses.keys():
                for dim in self.test_losses[loss][1].keys():
                    seed(self.seed_counter)
                    function_generator = generator.FunGen(dim, self.test_losses[loss][1][dim], loss)
                    functions = function_generator.generate(self.test_losses[loss][0])
                    for f in functions:
                        points = generator.generate_points(dim, self.m, self.radius, f[1])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if self.step > 0:
                                optimizer = tf.train.RMSPropOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate.astype(np.float32), global_step, self.step, self.dec, name='test_gradient'), decay=decay).minimize(f[0], global_step=global_step)
                            else:
                                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                temp = np.empty(self.iterations)
                                for i in range(self.iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                    temp[i] = f_curr

                                results = np.append(results, np.array([temp]), axis=0)
            return (metric_name, metric(results, axis=0))

        with Pool(processes) as p:
            pairs = [[best_params['best'][0], best_params['best'][1], np.min, 'best'], [best_params['worst'][0], best_params['worst'][1], np.max, 'worst'], [best_params['mean'][0], best_params['mean'][1], np.mean, 'mean'], [best_params['median'][0], best_params['median'][1], np.median, 'median']]
            test_results = p.map(test, pairs)
        test_results = {x: y for (x, y) in test_results}

        fig = plt.figure()
        plt.plot(self.x, test_results['best'], 'r', alpha=0.6, label='Best, LR: {}, DEC: {}'.format(str(round(best_params['best'][0], 5)), str(round(best_params['best'][1], 5))))
        plt.plot(self.x, test_results['worst'], 'm-o', alpha=0.3, label='Worst, LR: {}, DEC: {}'.format(str(round(best_params['worst'][0], 5)), str(round(best_params['worst'][1], 5))))
        plt.plot(self.x, test_results['mean'], 'g-*', alpha=0.3, label='Mean, LR: {}, DEC: {}'.format(str(round(best_params['mean'][0], 5)), str(round(best_params['mean'][1], 5))))
        plt.plot(self.x, test_results['median'], 'b--', alpha=0.6, label='Median, LR: {}, DEC: {}'.format(str(round(best_params['median'][0], 5)), str(round(best_params['median'][1], 5))))
        plt.title("RMSProb")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'rms_train.png', dpi=300)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'best', 'iteration': i, 'value': test_results['best'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'worst', 'iteration': i, 'value': test_results['worst'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'mean', 'iteration': i, 'value': test_results['mean'][i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'median', 'iteration': i, 'value': test_results['median'][i]})

        print('RMS testing done\n')
