import tensorflow as tf
from random import random
import numpy as np


class FunGen:
    def __init__(self, dim, n, scope):
        self.dim = dim
        self.n = n
        self.functions = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.x = tf.get_variable('x' + str(dim), [dim], )

    def generate(self, proto_function):
        self.functions = [proto_function(self.x, self.dim) for _ in range(self.n)]
        return self.functions


def generate_point(dim, r, start):
    angles = [0]
    for i in range(1, dim):
        angles.append(random() * 2 * np.pi)
    point = []
    for k in range(dim):
        temp = r * np.cos(angles[k])
        for a in range(k + 1, dim):
            temp *= np.sin(angles[a])
        point.append(temp)
    return [x + y for x, y in zip(point, start)]


def generate_points(dim, n, r, start):
    return [generate_point(dim, r, start) for _ in range(n)]

