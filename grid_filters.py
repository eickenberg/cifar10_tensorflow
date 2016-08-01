"""
Author: angles
Date and time: 23/07/16 - 17:25
"""
import tensorflow as tf
import numpy as np


def make_grid(weights):
    # The number of input channels has to be 1 or 3 (grayscale or RGB)
    # Notice that it shows all the filters only if the nb_filters is a perfect square
    nb_filters = weights.get_shape().dims[3]
    filters = tf.split(3, nb_filters, weights)
    for index in range(nb_filters):
        filters[index] = tf.pad(filters[index], [[1, 1], [1, 1], [0, 0], [0, 0]])
    side = int(np.sqrt(nb_filters.value))
    grid_rows = []
    for idx_row in range(side):
        grid_rows.append(tf.concat(0, filters[side * idx_row:side * (idx_row + 1)]))
    grid_transposed = tf.concat(1, grid_rows)
    grid = tf.transpose(grid_transposed, [3, 0, 1, 2])
    return grid


"""
# For testing with debugger:
weights_shape = [3, 3, 3, 64]
msr_init_std = np.sqrt(2 / (5 * 5 * 16))
# Microsoft Research weights initialization when using the ReLU as non-linearity
weights_init = tf.random_normal(weights_shape, mean=0, stddev=msr_init_std)
weights = tf.get_variable('weights', initializer=weights_init)
grid = make_grid(weights)
"""
