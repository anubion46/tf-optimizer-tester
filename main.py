import loss_functions as lf
import method_runner as mr
import numpy as np
import os
from multiprocessing import Pool
import time


def standart_test():
  # Set seed 
  seed_counter = 10
  # Iteration threshold
  iterations = 30
  # Number of points for functions to start from
  m = 1
  # Radius for points
  radius = 0.8

  # These 2 are for adaptive learning rates 
  step = 0
  dec = 0.1
  

  # Dictionary with functions (with corresponding dimensions and counts) that will be used to identify best parameters
  train_losses = {
                  'sum_sin': [lf.sum_sin, {15: 2, 25: 2, 35: 2}],
                  'sum_powers': [lf.sum_powers, {15: 2, 25: 2, 35: 2}],
                  'rastrigin': [lf.rastrigin, {15: 2, 25: 2, 35: 2}],
                  'deb01': [lf.deb01, {15: 2, 25: 2, 35: 2}],
                  'alpine01': [lf.alpine01, {15: 2, 25: 2, 35: 2}]
                  }

  # Dictionary with functions (with corresponding dimensions and counts) that will be used during testing
  test_losses = {
                 'sum_sin': [lf.sum_sin, {15: 3, 25: 3, 35: 3}],
                 'sum_powers': [lf.sum_powers, {15: 3, 25: 3, 35: 3}],
                 'rastrigin': [lf.rastrigin, {15: 3, 25: 3, 35: 3}],
                 'deb01': [lf.deb01, {15: 3, 25: 3, 35: 3}],
                 'alpine01': [lf.alpine01, {15: 3, 25: 3, 35: 3}]
                 }

  nspace = 10


  # Create directory with name given in dest if it doest not already exists
  dest = 'output/standart_test/'
  if not os.path.exists(dest):
      os.mkdir(dest)

  # # Create MethodRunner object with all the needed parameters 
  runner = mr.MethodRunner(seed_counter, iterations, m, radius, dec, 0, train_losses, test_losses, save=dest)

  runner.runGradientDescent(np.linspace(0.01, 0.5, nspace))
  runner.runAdam(np.linspace(0.01, 0.2, nspace))
  runner.runMomentum(np.linspace(0.01, 0.5, nspace))
  runner.runAdagrad(np.linspace(0.01,0.5, nspace))
  runner.runAdadelta(np.linspace(0.1, 10, nspace), np.linspace(0.9, 0.99, 2))
  runner.runRMS(np.linspace(0.01, 1.0, nspace), np.linspace(0.9, 0.99, 2))

def high_dim_test():
  # Set seed 
  seed_counter = 10
  # Iteration threshold
  iterations = 80
  # Number of points for functions to start from
  m = 2
  # Radius for points
  radius = 0.8

  # These 2 are for adaptive learning rates 
  step = 0
  dec = 0.1
  

  # Dictionary with functions (with corresponding dimensions and counts) that will be used to identify best parameters
  train_losses = {
                  'rastrigin': [lf.rastrigin, {500: 2, 1000: 2}],
                  'deb01': [lf.rastrigin, {500: 2, 1000: 2}],
                  'alpine01': [lf.rastrigin, {500: 2, 1000: 2}]
                  }

  # Dictionary with functions (with corresponding dimensions and counts) that will be used during testing
  test_losses = {
                  'rastrigin': [lf.rastrigin, {500: 4, 1000: 4}],
                  'deb01': [lf.rastrigin, {500: 4, 1000: 4}],
                  'alpine01': [lf.rastrigin, {500: 4, 1000: 4}]
                }

  nspace = 10


  # Create directory with name given in dest if it doest not already exists
  dest = 'output/high_dim_test/'
  if not os.path.exists(dest):
      os.mkdir(dest)

  # # Create MethodRunner object with all the needed parameters 
  runner = mr.MethodRunner(seed_counter, iterations, m, radius, dec, step, train_losses, test_losses, save=dest)

  runner.runGradientDescent(np.geomspace(0.001, 0.5, nspace))
  runner.runAdam(np.geomspace(0.01, 0.3, nspace))
  # runner.runMomentum(np.linspace(0.001, 0.2, nspace))
  # runner.runAdagrad(np.linspace(0.1, 1.0, nspace))
  # runner.runAdadelta(np.linspace(15, 50, nspace), np.linspace(0.9, 0.99, 2))
  # runner.runRMS(np.linspace(0.01, 1.0, nspace), np.linspace(0.9, 0.99, 2))

  # Show plots when all of them are ready
  # runner.show_plot()


def main():
  # Dont show warnings and other system information
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  # Set a timer when we start running the code
  timer = time.time()

  high_dim_test()

  # Print how much time it took to execute the code
  print('Time took: {}'.format(time.time() - timer))
  

if __name__ == '__main__':
  main()
