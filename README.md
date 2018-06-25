## What is this thing?

This is a simple tester for different optimization methods (such as Adam, Adagrad and so on) that are present in the **Tensorflow** library. It runs tests using given test multi-dimensional functions to identify best arguments for each method. It's not fully automated so the program provides usefull data that can be used to identify the best method among tested (such as convergence plots and datasets).

## What is the point?

The tester works with given funcions in conjunction using 4 metrics: 
* *best possible result during single epoch*; 
* *worst possible result during single epoch*; 
* *mean of results during single epoch*, 
* *median of results during single epoch*. 

User may add own functions and run prefered tests to see which method is better in particular. 


## What was it built with?

```
* Python 3+
* Tensorflow 1.5+
* Numpy
* Matplotlib
* Pathos

* R (was used separately)
```
