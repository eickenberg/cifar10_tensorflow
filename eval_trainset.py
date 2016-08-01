"""
Author: angles
Date and time: 27/07/16 - 18:22
"""
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from batch_functions import provide_batch
from szagoruyko import inference

# Creating batches for training and testing
batch_size = 128
with tf.name_scope('batch_testing_train'):
    batch_images_train_tensor, batch_labels_train_tensor = provide_batch('train', batch_size, training=False)

# Predicting the labels using our inference function
with tf.variable_scope('inference') as scope:
    logits_testing_train_tensor = inference(batch_images_train_tensor, is_training=False)

with tf.name_scope('accuracy'):
    top1_train_tensor = tf.nn.in_top_k(logits_testing_train_tensor, batch_labels_train_tensor, 1)

saver = tf.train.Saver()

# Create a session for running operations in the Graph
######################################################
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Initialize the variables.
sess.run(tf.initialize_all_variables())
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# Saving the graph to visualize in Tensorboard
summary_writer = tf.train.SummaryWriter('./log', sess.graph)
merged_summary_operation = tf.merge_all_summaries()
# The main loop
###############
nb_train_samples = 50000
nb_batches_per_epoch_train = int(nb_train_samples / batch_size)
try:
    # If we want to continue training a model:
    saver.restore(sess, "./model.ckpt")
    merged_summary = sess.run(merged_summary_operation)
    summary_writer.add_summary(merged_summary)
    # Evaluating training set
    accuracy_epoch = 0
    print(datetime.now(), end=' ')
    for index_batch in range(nb_batches_per_epoch_train):
        if (index_batch + 1) % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        top1_train = sess.run([top1_train_tensor])
        accuracy_epoch += np.sum(top1_train)
    accuracy_epoch /= nb_batches_per_epoch_train * batch_size
    print(' Total training accuracy: %f' % accuracy_epoch)
except tf.errors.OutOfRangeError:
    print('Something happened with the queue runners')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
