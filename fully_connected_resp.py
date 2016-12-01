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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path
import time
import pdb
import sys


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn import metrics
import define_net_resp



# Basic model parameters as external flags.
batch_size=int(sys.argv[4])
hidden1=10
hidden2=10
hidden3=10
learning_rate=0.1
max_steps=100

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    data_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  data_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         define_net_resp.FEATURES))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return data_placeholder, labels_placeholder


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  #file_train = "train_sub2.csv"
  #file_test= "test_sub2.csv"
  file_train=str(sys.argv[1])
  file_test=str(sys.argv[2])
    # setup text reader
  l_examples=file_len(file_train)
  l_examples_test=file_len(file_test)
  filename_queue = tf.train.string_input_producer([file_train],shuffle=False)
  reader = tf.TextLineReader(skip_header_lines=0)
  _, csv_row = reader.read(filename_queue)
  filename_queue_test = tf.train.string_input_producer([file_test],shuffle=False)
  reader = tf.TextLineReader(skip_header_lines=0)
  _, csv_row_test = reader.read(filename_queue_test)
  if (int(sys.argv[5])==0):
    record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14 = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13])
    col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t = tf.decode_csv(csv_row_test, record_defaults=record_defaults)
    features_test = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t])
  elif (int(sys.argv[5])==1):
    record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],[1.], [1.],[1.], [1.],[1.], [1.], [1.], [1]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14, col15,  col16,  col17, col18, col19, col20, col21 = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14, col15,  col16,  col17, col18, col19, col20])
    col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t, col15t,  col16t,  col17t, col18t, col19t, col20t, col21t = tf.decode_csv(csv_row_test, record_defaults=record_defaults)
    features_test = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t, col15t,  col16t,  col17t, col18t, col19t, col20t])
  else:
    record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],[1.],[1.],[1.],[1]]
    col1, col2,col3,col4, col5, col6, col7, col8, col9, col10,col11,col12,col13,col14,col15,col16,col17 = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([col1,col2,col3,col4, col5,col6,col7, col8,col9, col10,col11,col12, col13,col14,col15,col16])
    col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t, col15t,  col16t,  col17t = tf.decode_csv(csv_row_test, record_defaults=record_defaults)
    features_test = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t, col15t, col16t])
  min_after_dequeue = 10000000
  capacity = min_after_dequeue +  define_net_resp.FEATURES * batch_size
  
  true_count = 0  # Counts the number of correct predictions.
  if (sys.argv[6]==0):
    steps_per_epoch = l_examples_test // batch_size
    p_num_examples = steps_per_epoch * batch_size
  else:
      steps_per_epoch = l_examples_test // 1
      p_num_examples = steps_per_epoch * 1

  data_placeholder, labels_placeholder = placeholder_inputs(batch_size)
  data_placeholder_tt, labels_placeholder_tt = placeholder_inputs(1)
  if (int(sys.argv[5])==0):
    images_batch,label_batch = tf.train.batch([features,col14-1],
                                                batch_size=batch_size,
                                                capacity=capacity,num_threads=1)
    if (sys.argv[6]==0):
      example_batch, label_batch_test = tf.train.batch([features_test,col14t-1], batch_size=batch_size, capacity=capacity,num_threads=1)
    else:
      example_batch, label_batch_test = tf.train.batch([features_test,col14t-1], batch_size=batch_size, capacity=capacity,num_threads=1)
  elif (int(sys.argv[5])==1):
    images_batch,label_batch = tf.train.batch([features,col21-1],
                                              batch_size=batch_size,
                                              capacity=capacity,num_threads=1)
    if (sys.argv[6]==0):
      example_batch, label_batch_test = tf.train.batch([features_test,col21t-1], batch_size=batch_size, capacity=capacity,num_threads=1)
    else:
      example_batch, label_batch_test = tf.train.batch([features_test,col21t-1], batch_size=1, capacity=capacity,num_threads=1)
  else:
    images_batch,label_batch = tf.train.batch([features,col17-1],
                                              batch_size=batch_size,
                                              capacity=capacity,num_threads=1)
    if (sys.argv[6]==0):
      example_batch, label_batch_test = tf.train.batch([features_test,col17t-1], batch_size=batch_size, capacity=capacity,num_threads=1)
    else:
      example_batch, label_batch_test = tf.train.batch([features_test,col17t-1], batch_size=1, capacity=capacity,num_threads=1)

  logits = define_net_resp.inference(data_placeholder,hidden1,
                                           hidden2,hidden3)
  eval_correct = define_net_resp.evaluation(logits, labels_placeholder)
  loss = define_net_resp.loss(logits, labels_placeholder)
  train_op = define_net_resp.training(loss,learning_rate)
  init = tf.initialize_all_variables()
  with tf.Session() as sess:
      sess.run(init)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      #training process
      for step in xrange(max_steps):
        start_time = time.time()
        example, label = sess.run([images_batch, label_batch])
        _, loss_value = sess.run([train_op, loss],feed_dict={data_placeholder:example,labels_placeholder:label})
        duration = time.time() - start_time
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      temp=0.0
      pres=0.0
      rec=0.0
      #test process
      for step in xrange(steps_per_epoch):
        print(true_count,steps_per_epoch)
        ex, lab = sess.run([example_batch, label_batch_test])
        feed_dict={data_placeholder: ex,labels_placeholder:lab}
        print(true_count,steps_per_epoch)
        [true_count,y_d]= sess.run([eval_correct,logits], feed_dict=feed_dict)
        temp+=true_count
        print(lab,np.argmax(y_d,1))
        a_c=np.concatenate((np.transpose([lab]),np.transpose([np.argmax(y_d,1)])),axis=1)
        if step==0:
              temp_c=a_c
        else:
              print(temp_c,a_c)
              temp_c=np.vstack((temp_c,a_c))
        print(a_c)
        pres+=metrics.precision_score(lab,np.argmax(y_d,1),average='weighted')
        rec+=metrics.recall_score(lab,np.argmax(y_d,1),average='weighted')
      precision = temp / steps_per_epoch
      prec = pres / steps_per_epoch
      recrec= rec / steps_per_epoch
      #np.savetxt("cs_testing_sub2.csv", temp_c, delimiter=",",fmt="%d")
      np.savetxt(str(sys.argv[3]), temp_c, delimiter=",",fmt="%d")
      print('  Num examples: %d   Validation Accuracy @ 1: %0.04f Precision @ 1: %0.04f Recall @ 1: %0.04f ' %
          (p_num_examples,precision,prec,recrec))
                                                  #print "Recall",  metrics.recall_score(y_true, y_pred)
                                                  #print "f1_score", metrics.f1_score(y_true, y_pred)
      coord.request_stop()
      coord.join(threads)
      print("\ndone loading")



def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
