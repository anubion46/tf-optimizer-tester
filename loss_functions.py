import tensorflow as tf
from random import random
import numpy as np


def sum_powers(x, dim):
    a = np.float32(np.random.random(dim))
    min_point = np.float32(np.random.random(dim))
    f = tf.reduce_sum(a * (x - min_point) ** 2)
    function_look = '-'
    return f, function_look, min_point, 0.0


def sum_sin(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = tf.abs(tf.reduce_sum(a * sin(x))) / n
    function_look = 'sum of a_i * sin(x_i) divided by n'
    return f, function_look, np.zeros(dim, dtype=np.float32), 0.0


# def csendes(x, dim):
#     f = tf.reduce_sum(x**6 * (2.0 + tf.sin(1.0/x)))
#     function_look = 'sum of xi^6 * (2 + sin(1/xi)) from 1 to ' + str(dim)
#     return f, function_look, np.zeros(dim, dtype=np.float32), 0.0


def rastrigin(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = 10.0 * dim * tf.reduce_sum(a * (x ** 2 - 10.0 * tf.cos(2.0 * np.pi * x) + 10.0))
    function_look = '-'
    return f, function_look, np.zeros(dim, dtype=np.float32), 0.0


def deb01(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = 1.0/dim * tf.reduce_sum(a * tf.sin(5.0 * np.pi * x) ** 6)
    function_look = '-'
    return f, function_look, np.zeros(dim, dtype=np.float32), 0.0


def alpine01(x, dim):
    a = tf.constant(np.float32(np.random.random(dim)))
    f = tf.reduce_sum(a * tf.abs(x * tf.sin(x) + 0.1 * x))
    function_look = '-'
    return f, function_look, np.zeros(dim, dtype=np.float32), 0.0
