from random import random, seed
import generator
import tensorflow as tf
import csv


def monte_carlo_local(funcs, f, dim, amount, window):
    seed(1234)
    k = amount / window
    points = [[random() for _ in range(dim)] for _ in range(amount)]
    with tf.Session() as sess:
        old = sess.run(f[0], feed_dict={funcs.x: points[0]})
        index = 0
        min_values = []
        for i in range(1, len(points)):
            new = sess.run(f[0], feed_dict={funcs.x: points[i]})
            if old > new:
                old = new
                index = i
            if (i + 2) % k == 0:
                min_values.append(new)

    return old, points[index], min_values


def monte_carlo(functions, dim, amount, window):
    temp = [monte_carlo_local(functions, f, dim, amount, window) for f in functions.generate('c2')]
    for i in range(len(temp)):
        with open('output/mc_' + str(i) + '.csv', 'w', newline='') as output:
            # Column names for tests files (i.e. c1, c2)
            fieldnames = ['value']
            # Writing columns to file. Look up python's csv module
            thewriter = csv.DictWriter(output, fieldnames=fieldnames)
            thewriter.writeheader()
            for value in temp[i][2]:
                thewriter.writerow({'value': value})


amount = 15000
dim = 3
window = 100
n = 1
functions = generator.FunGen(dim, n)
print(monte_carlo(functions, dim, amount, window))
