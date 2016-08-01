"""
Author: angles
Date and time: 30/07/16 - 19:24
"""

"""
Author: angles
Date and time: 22/07/16 - 16:24
"""
import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops

from grid_filters import make_grid


def batch_norm(inputs, decay=0.9, center=True, scale=True, epsilon=0.001, is_training=True):
    inputs_shape = inputs.get_shape()
    dtype = inputs.dtype.base_dtype
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    beta, gamma = None, None
    if center:
        beta = tf.get_variable('beta', shape=params_shape, dtype=dtype, initializer=init_ops.zeros_initializer)
    if scale:
        gamma = tf.get_variable('gamma', shape=params_shape, dtype=dtype, initializer=init_ops.ones_initializer)
    moving_mean = tf.get_variable('moving_mean', shape=params_shape, dtype=dtype,
                                  initializer=init_ops.zeros_initializer, trainable=False)
    moving_variance = tf.get_variable('moving_variance', shape=params_shape, dtype=dtype,
                                      initializer=init_ops.ones_initializer, trainable=False)
    if is_training:
        # Calculate the moments based on the individual batch.
        mean, variance = nn.moments(inputs, axis, shift=moving_mean)
        # Update the moving_mean and moving_variance moments.
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
        # Make sure the updates are computed here.
        with ops.control_dependencies([update_moving_mean, update_moving_variance]):
            outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
    else:
        outputs = nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())
    return outputs


def convolution(inputs, k, nb_channels_output, downsampling=False, is_training=True):
    dtype = inputs.dtype.base_dtype
    nb_channels_input = utils.last_dimension(inputs.get_shape(), min_rank=4)
    kernel_h, kernel_w = utils.two_element_tuple(k)
    weights_shape = [kernel_h, kernel_w, nb_channels_input, nb_channels_output]
    msr_init_std = np.sqrt(2 / (kernel_h * kernel_w * nb_channels_output))
    weights_init = tf.random_normal(weights_shape, mean=0, stddev=msr_init_std, dtype=dtype)
    weights = tf.get_variable('weights', initializer=weights_init)
    # tf.add_to_collection('losses', tf.nn.l2_loss(weights))
    if (nb_channels_input == 1) or (nb_channels_input == 3):
        grid = make_grid(weights)
        tf.image_summary(tf.get_default_graph().unique_name('Filters', mark_as_used=False), grid)
    if (downsampling):
        conv = tf.nn.conv2d(inputs, weights, [1, 2, 2, 1], padding='SAME')
    else:
        conv = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
    return conv


def linear(name_scope, inputs, nb_output_channels):
    with tf.variable_scope(name_scope):
        dtype = inputs.dtype.base_dtype
        nb_input_channels = utils.last_dimension(inputs.get_shape(), min_rank=2)
        weights_shape = [nb_input_channels, nb_output_channels]
        weights = tf.get_variable('weights', weights_shape, initializer=initializers.xavier_initializer(), dtype=dtype)
        # tf.add_to_collection('losses', tf.nn.l2_loss(weights))
        output = tf.matmul(inputs, weights)
        bias = tf.get_variable('bias', [nb_output_channels], initializer=init_ops.zeros_initializer, dtype=dtype)
        output = nn.bias_add(output, bias)
    return output


def layer(name_scope, input, k, nb_channels_output, downsampling=False, is_training=True):
    with tf.variable_scope(name_scope):
        output = convolution(input, k, nb_channels_output, downsampling, is_training)
        output = batch_norm(output, is_training=is_training)
        output = nn.relu(output)
    return output


def group(name_scope, input, nb_channels_output, nb_repetitions, k, downsampling=False, is_training=True):
    with tf.variable_scope(name_scope):
        output = layer('Conv1', input, k, nb_channels_output, downsampling, is_training=is_training)
        if nb_repetitions > 1:
            for repetition in range(nb_repetitions - 1):
                temp_name_scope = 'Conv' + str(repetition + 2)
                output = layer(temp_name_scope, output, k, nb_channels_output, is_training=is_training)
    return output


def inference(input, is_training=True):
    output = group('group1', input, 16, 1, 5, is_training=is_training)
    output = group('group2', output, 32, 1, 5, downsampling=True, is_training=is_training)
    output = group('group3', output, 64, 4, 3, is_training=is_training)
    output = group('group4', output, 128, 4, 3, downsampling=True, is_training=is_training)
    output = nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
    output = tf.squeeze(output, squeeze_dims=[1, 2])
    output = linear('linear', output, 10)
    return output


"""
# For testing with debugger:
input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
output = inference(input)
"""
