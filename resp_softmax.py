# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import pdb

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

NUM_FEATURES = 13
NUM_LABELS = 4
batch_size=10

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def running():
  
  filename = "train_sub4.csv"
  file_name_test= "test_sub4.csv"
  # setup text reader
  file_length = file_len(filename)
  filename_queue = tf.train.string_input_producer([filename],shuffle=False)
  reader = tf.TextLineReader(skip_header_lines=0)
  _, csv_row = reader.read(filename_queue)
  filename_queue_test = tf.train.string_input_producer([file_name_test],shuffle=False)
  reader = tf.TextLineReader(skip_header_lines=0)
  _, csv_row_test = reader.read(filename_queue_test)


  # Default values, in case of empty columns. Also specifies the type of the
  # decoded result.
  record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]
  col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14 = tf.decode_csv(csv_row, record_defaults=record_defaults)
  features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13])
  col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t = tf.decode_csv(csv_row_test, record_defaults=record_defaults)
  features_test = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t])


  # Create the model
  x = tf.placeholder(tf.float32, [None,NUM_FEATURES])
  W = tf.Variable(tf.zeros([NUM_FEATURES, NUM_LABELS]))
  b = tf.Variable(tf.zeros([NUM_LABELS]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  min_after_dequeue = 1000
  capacity = min_after_dequeue + NUM_FEATURES * batch_size

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

   #sess = tf.InteractiveSession()
  # Train
  images_batch,label_batch = tf.train.batch([features,col14-1],
                                          batch_size=batch_size,
                                          capacity=capacity)
  example_batch, label_batch_test = tf.train.batch([features_test,col14t-1], batch_size=batch_size, capacity=capacity)
  ll=tf.to_float(tf.one_hot(tf.to_int64(label_batch),4,1,0))
  lt=tf.to_float(tf.one_hot(tf.to_int64(label_batch_test),4,1,0))
  init=tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in xrange(500):
        start_time = time.time()
        example, label = sess.run([images_batch, ll])
        sess.run(train_step, feed_dict={x: example, y_: label})

  # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: example_batch.eval() ,y_: lt.eval()}))
    coord.request_stop()
    coord.join(threads)

def main(_):
    running()

if __name__ == '__main__':
  tf.app.run()
