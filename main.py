import os
import time

import numpy as np

import loss_functions as lf
import method_runner as mr


def high_dim_test():
    # Set seed
    seed_counter = 20
    # Iteration threshold
    iterations = 60
    # Number of points for functions to start from
    m = 2
    # Radius for points
    radius = 0.8

    # These 2 are for adaptive learning rates
    step = 0
    dec = 0.1

    # Dictionary with functions (with corresponding dimensions and counts) that will be used to identify best parameters
    train_losses = {
        'rastrigin': [lf.rastrigin, {5000: 1, 10000: 1}],
        'deb01': [lf.rastrigin, {5000: 1, 10000: 1}],
        'alpine01': [lf.rastrigin, {5000: 1, 10000: 1}]
    }

    # Dictionary with functions (with corresponding dimensions and counts) that will be used during testing
    test_losses = {
        'rastrigin': [lf.rastrigin, {5000: 3, 10000: 3}],
        'deb01': [lf.rastrigin, {5000: 3, 10000: 3}],
        'alpine01': [lf.rastrigin, {5000: 3, 10000: 3}]
    }

    nspace = 15

    # Create directory with name given in dest if it doest not already exists
    dest = 'output/high_dim_test/'
    if not os.path.exists(dest):
        os.mkdir(dest)

    # # Create MethodRunner object with all the needed parameters
    runner = mr.MethodRunner(seed_counter, iterations, m, radius, dec, step, train_losses, test_losses, save=dest)

    runner.runGradientDescent(np.geomspace(0.00001, 0.001, nspace))
    runner.runAdam(np.geomspace(0.001, 0.03, nspace))
    runner.runMomentum(np.geomspace(0.0001, 0.1, nspace))
    runner.runAdagrad(np.geomspace(0.2, 3.0, nspace))
    runner.runAdadelta(np.geomspace(4.0, 7.0, nspace), np.linspace(0.9, 0.99, 2))
    runner.runRMS(np.geomspace(0.001, 0.1, nspace), np.linspace(0.9, 0.99, 2))

    # Show plots when all of them are ready
    # runner.show_plot()


def main():
    # Don't show warnings and other system information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set a timer when we start running the code
    timer = time.time()

    # Start your test
    high_dim_test()

    # Print how much time it took to execute the code
    print('Time took: {}'.format(time.time() - timer))


if __name__ == '__main__':
    main()
