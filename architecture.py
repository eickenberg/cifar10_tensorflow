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
from haar import tree_conv

from grid_filters import make_grid


def batch_norm(name_scope, inputs, decay=0.9, epsilon=0.001, is_training=True):
    with tf.variable_scope(name_scope):
        inputs_shape = inputs.get_shape()
        dtype = inputs.dtype.base_dtype
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        beta = tf.get_variable('beta', shape=params_shape, dtype=dtype, initializer=init_ops.zeros_initializer)
        gamma = tf.get_variable('gamma', shape=params_shape, dtype=dtype, initializer=init_ops.ones_initializer)
        moving_mean = tf.get_variable('moving_mean', shape=params_shape, dtype=dtype,
                                      initializer=init_ops.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable('moving_variance', shape=params_shape, dtype=dtype,
                                          initializer=init_ops.ones_initializer, trainable=False)
        if is_training:
            mean, variance = nn.moments(inputs, axis, shift=moving_mean)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            with ops.control_dependencies([update_moving_mean, update_moving_variance]):
                outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        else:
            outputs = nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
    return outputs


def convolution(name_scope, inputs, nb_channels_output, k, s, p):
    with tf.variable_scope(name_scope):
        dtype = inputs.dtype.base_dtype
        nb_channels_input = utils.last_dimension(inputs.get_shape(), min_rank=4)
        kernel_h, kernel_w = utils.two_element_tuple(k)
        weights_shape = [kernel_h, kernel_w, nb_channels_input, nb_channels_output]
        msr_init_std = np.sqrt(2 / (kernel_h * kernel_w * nb_channels_output))
        weights_init = tf.random_normal(weights_shape, mean=0, stddev=msr_init_std, dtype=dtype)
        weights = tf.get_variable('weights', initializer=weights_init)
        if (nb_channels_input == 1) or (nb_channels_input == 3):
            grid = make_grid(weights)
            tf.image_summary(tf.get_default_graph().unique_name('Filters', mark_as_used=False), grid)
        padded_inputs = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]], 'CONSTANT')
        outputs = tf.nn.conv2d(padded_inputs, weights, [1, s, s, 1], padding='VALID')
    return outputs

def tree_convolution(name_scope, inputs, nb_channels_output, k, s, p):
    with tf.variable_scope(name_scope):
        dtype = inputs.dtype.base_dtype
        nb_channels_input = utils.last_dimension(inputs.get_shape(), min_rank=4)
        kernel_h, kernel_w = utils.two_element_tuple(k)
        weights_shape = [kernel_h, kernel_w, nb_channels_input, nb_channels_output]
        msr_init_std = np.sqrt(2 / (kernel_h * kernel_w * nb_channels_output))
        weights_init = tf.random_normal(weights_shape, mean=0, stddev=msr_init_std, dtype=dtype)
        weights = tf.get_variable('weights', initializer=weights_init)
        if (nb_channels_input == 1) or (nb_channels_input == 3):
            grid = make_grid(weights)
            tf.image_summary(tf.get_default_graph().unique_name('Filters', mark_as_used=False), grid)
        padded_inputs = tf.pad(inputs, [[0, 0], [p, p], [p, p], [0, 0]], 'CONSTANT')
        outputs = tree_conv(padded_inputs, weights, strides=(s, s), padding='VALID')
    return outputs


def linear(name_scope, inputs, nb_output_channels):
    with tf.variable_scope(name_scope):
        dtype = inputs.dtype.base_dtype
        nb_input_channels = utils.last_dimension(inputs.get_shape(), min_rank=2)
        weights_shape = [nb_input_channels, nb_output_channels]
        weights = tf.get_variable('weights', weights_shape, initializer=initializers.xavier_initializer(), dtype=dtype)
        output = tf.matmul(inputs, weights)
        bias = tf.get_variable('bias', [nb_output_channels], initializer=init_ops.zeros_initializer, dtype=dtype)
        output = nn.bias_add(output, bias)
    return output


def layer(name_scope, inputs, nb_channels_output, k, s, p, is_training=True):
    with tf.variable_scope(name_scope):
        outputs = convolution('convolution', inputs, nb_channels_output, k, s, p)
        outputs = batch_norm('bn', outputs, is_training=is_training)
        outputs = nn.relu(outputs)
    return outputs

def tree_layer(name_scope, inputs, nb_channels_output, k, s, p, is_training=True):
    with tf.variable_scope(name_scope):
        outputs = tree_convolution('tree_conv', inputs, nb_channels_output, k, s, p)
        outputs = batch_norm('bn', outputs, is_training=is_training)
        outputs = nn.relu(outputs)
        print ('tree layer here')
    return outputs


def inference(inputs, is_training):
    outputs = tree_layer('l1', inputs, nb_channels_output=36, k=5, s=1, p=2, is_training=is_training)
    # print(outputs.get_shape())
    # outputs = layer('l1', inputs, nb_channels_output=36, k=5, s=1, p=2, is_training=is_training)
    # print(outputs.get_shape())
    outputs = tree_layer('l2', outputs, nb_channels_output=64, k=5, s=1, p=2, is_training=is_training)
    outputs = tree_layer('l3', outputs, nb_channels_output=36, k=2, s=2, p=0, is_training=is_training)
    outputs = tree_layer('l4', outputs, nb_channels_output=72, k=3, s=1, p=1, is_training=is_training)
    outputs = layer('l5', outputs, nb_channels_output=128, k=3, s=1, p=1, is_training=is_training)
    outputs = layer('l6', outputs, nb_channels_output=72, k=2, s=2, p=0, is_training=is_training)
    outputs = nn.avg_pool(outputs, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
    outputs = tf.squeeze(outputs, squeeze_dims=[1, 2])
    outputs = linear('linear', outputs, 10)
    return outputs


"""
def group(name_scope, input, nb_channels_output, nb_repetitions, k, downsampling=False, is_training=True):
    with tf.variable_scope(name_scope):
        output = layer('Conv1', input, k, nb_channels_output, downsampling, is_training=is_training)
        if nb_repetitions > 1:
            for repetition in range(nb_repetitions - 1):
                temp_name_scope = 'Conv' + str(repetition + 2)
                output = layer(temp_name_scope, output, k, nb_channels_output, is_training=is_training)
    return output

def inference(input, is_training=True):
    output = group('group1', input, 64, 1, 5, is_training=is_training)
    output = group('group2', output, 64, 1, 5, downsampling=True, is_training=is_training)
    output = group('group3', output, 128, 4, 3, is_training=is_training)
    output = group('group4', output, 128, 4, 3, downsampling=True, is_training=is_training)
    output = nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
    output = tf.squeeze(output, squeeze_dims=[1, 2])
    output = linear('linear', output, 10)
    return output
"""

"""
# For testing with debugger:
def inference_temp(input, is_training=True):
    output = layer('one_layer', input, 3, 10, is_training=is_training)
    return output
inputs = tf.placeholder(dtype=tf.float32, shape=[128, 32, 32, 3])
output = inference(inputs, is_training=True)
"""
