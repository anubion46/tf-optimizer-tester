import tensorflow as tf
from random import random
import numpy as np


"""
This file contains testing functions. 

To add a testing function define a new python function with arguments:
- x: tensroflow vector; 
- dim: dimension of x;

Created python function should return:
1. tensorflow vector-function
2. a list of minimum coordinates

It is highly advised for testing function to have minimum in 0. 
"""
def sum_powers(x, dim):
    a = np.float32(np.random.random(dim))
    min_point = np.float32(np.random.random(dim))
    f = tf.reduce_sum(a * (x - min_point) ** 2)
    function_look = '-'
    return f, min_point


def sum_sin(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = tf.reduce_sum(a - tf.abs(a * tf.sin(x))) / dim
    function_look = 'sum of a_i * sin(x_i) divided by n'
    return f, np.zeros(dim, dtype=np.float32)


def deb01(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = 1.0/dim * tf.reduce_sum(a * tf.sin(5.0 * np.pi * x) ** 6)
    function_look = '-'
    return f, np.zeros(dim, dtype=np.float32)


def alpine01(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = tf.reduce_sum(a * tf.abs(x * tf.sin(x) + 0.1 * x))
    function_look = '-'
    return f, np.zeros(dim, dtype=np.float32)


def rastrigin(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = 10.0 * dim * tf.reduce_sum(a * (x ** 2 - 10.0 * tf.cos(2.0 * np.pi * x) + 10.0))
    function_look = '-'
    return f, np.zeros(dim, dtype=np.float32)