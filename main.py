"""
Author: ta
Date and time: 21/07/16 - 14:36
"""
import sys
import numpy as np
import tensorflow as tf
from batch_functions import provide_batch
from architecture_ed import inference
from datetime import datetime

# All the parameters:
nb_train_samples = 50000
nb_test_samples = 10000
nb_epochs_train = 5000
grayscale = False
batch_size = 128

optimizer_name = 'Adam'
momentum = 0.9
initial_learning_rate = 0.01
decay_rate = 0.2
nb_epochs_decay_learning_rate = 25
weights_decay = 0

restore_saved_model = False
dir_logs = './logs'
per_process_gpu_memory_fraction = 0.2

txt_file = open('results.txt', 'w')
txt_file.write('Training a DCN on CIFAR-10\n')
if grayscale:
    txt_file.write('Using Gray-Scale CIFAR-10\n')
txt_file.write('nb_train_samples: %i\n' % nb_train_samples)
txt_file.write('nb_test_samples: %i\n' % nb_test_samples)
txt_file.write('nb_epochs_train: %i\n' % nb_epochs_train)
txt_file.write('batch_size: %i\n' % batch_size)
txt_file.write('optimizer: %s\n' % optimizer_name)
txt_file.write('momentum(if using Momentum): %f\n' % momentum)
txt_file.write('initial_learning_rate: %f\n' % initial_learning_rate)
txt_file.write('decay_rate: %f\n' % decay_rate)
txt_file.write('nb_epochs_decay_learning_rate: %i\n' % nb_epochs_decay_learning_rate)
txt_file.write('weights_decay: %f\n' % weights_decay)
txt_file.write('Restore?: %i\n' % restore_saved_model)
txt_file.write('per_process_gpu_memory_fraction: %f\n' % per_process_gpu_memory_fraction)
txt_file.write('dir_logs: %s\n' % dir_logs)
txt_file.close()

# Creating batches for training and testing
with tf.name_scope('batch_training'):
    batch_images_training_tensor, batch_labels_training_tensor = provide_batch('train', batch_size, training=True,
                                                                               grayscale=grayscale)
with tf.name_scope('batch_testing_test'):
    batch_images_test_tensor, batch_labels_test_tensor = provide_batch('test', batch_size, training=False,
                                                                       grayscale=grayscale)

# Predicting the labels using our inference function
with tf.variable_scope('inference') as scope:
    logits_training_tensor = inference(batch_images_training_tensor, is_training=True)
    scope.reuse_variables()
    logits_testing_test_tensor = inference(batch_images_test_tensor, is_training=False)

# Computing the total loss
with tf.name_scope('compute_loss'):
    batch_labels_tensor = tf.cast(batch_labels_training_tensor, tf.int64)
    # Notice that the cross entropy function takes predictions as a vector with nb_labels coordinates
    # and the labels as a number between 0 and nb_labels-1
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_training_tensor, batch_labels_tensor)
    total_loss = tf.reduce_mean(cross_entropy)
    if weights_decay > 0:
        weights_norm = weights_decay * tf.reduce_sum(tf.pack([tf.nn.l2_loss(w) for w in tf.trainable_variables()]))
        total_loss = tf.add(total_loss, weights_norm)
    tf.scalar_summary('total_loss', total_loss)

nb_batches_per_epoch_train = int(nb_train_samples / batch_size)
global_step = tf.Variable(0, trainable=False)

with tf.name_scope('optimizer'):
    # Defining the optimizer to use in order to update the weights
    decay_steps = nb_epochs_decay_learning_rate * nb_batches_per_epoch_train
    learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate, global_step=global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=decay_rate, staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)
    if optimizer_name is 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    if optimizer_name is 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        print('Define optimizer')
        sys.exit()
    # Creating the operation that allows us to update the trainable variables
    # First we control the fact that before minimizing we should compute the total_loss
    with tf.control_dependencies([total_loss]):
        update_variables = optimizer.minimize(total_loss, global_step)

with tf.name_scope('accuracy'):
    top1_test_tensor = tf.nn.in_top_k(logits_testing_test_tensor, batch_labels_test_tensor, 1)

# Add histograms for trainable variables.
for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

saver = tf.train.Saver()

# Create a session for running operations in the Graph
######################################################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Initialize the variables.
sess.run(tf.initialize_all_variables())
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# Saving the graph to visualize in Tensorboard
summary_writer = tf.train.SummaryWriter(dir_logs, sess.graph)
merged_summary_operation = tf.merge_all_summaries()

# The main loop
###############
nb_batches_per_epoch_test = int(nb_test_samples / batch_size)
nb_training_steps = nb_epochs_train * nb_batches_per_epoch_train
try:
    # If we want to continue training a model:
    if restore_saved_model:
        saver.restore(sess, "./model.ckpt")
    epoch = 0
    print(datetime.now(), end=' ')
    for step in range(nb_training_steps):
        # Training
        sess.run(update_variables)
        merged_summary = sess.run(merged_summary_operation)
        summary_writer.add_summary(merged_summary, step)
        if (step + 1) % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        if (step + 1) % nb_batches_per_epoch_train == 0:
            merged_summary = sess.run(merged_summary_operation)
            summary_writer.add_summary(merged_summary, step)
            txt_file = open('results.txt', 'a')
            epoch += 1
            print(' Epoch %i' % epoch, end='')
            txt_file.write('%s Epoch %i' % (datetime.now(), epoch))
            # Evaluating test set
            accuracy_epoch = 0
            for index_batch in range(nb_batches_per_epoch_test):
                top1_test = sess.run([top1_test_tensor])
                accuracy_epoch += np.sum(top1_test)
            accuracy_epoch /= nb_batches_per_epoch_test * batch_size
            print(' Total test accuracy: %f \n' % accuracy_epoch)
            txt_file.write(' Total test accuracy: %f\n' % accuracy_epoch)
            saver.save(sess, "./model.ckpt")
            txt_file.close()
            print(datetime.now(), end=' ')
except tf.errors.OutOfRangeError:
    print('Something happened with the queue runners')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
