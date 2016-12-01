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
        file_train=str(sys.argv[1])
        # setup text reader
        l_examples=file_len(file_train)
        filename_queue = tf.train.string_input_producer([file_train],shuffle=False)
        reader = tf.TextLineReader(skip_header_lines=0)
        _, csv_row = reader.read(filename_queue)

        if (int(sys.argv[2])==0):
            record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1]]
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14 = tf.decode_csv(csv_row, record_defaults=record_defaults)
            features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13])
            col_t=tf.cast(tf.pack([col14]),tf.int32)
        elif (int(sys.argv[2])==1):
            record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],[1.], [1.],[1.], [1.],[1.], [1.], [1.], [1]]
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14, col15,  col16,  col17, col18, col19, col20, col21 = tf.decode_csv(csv_row, record_defaults=record_defaults)
            features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14, col15,  col16,  col17, col18, col19, col20])
            col_t=tf.cast(tf.pack([col21]),tf.int32)
        else:
            record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],[1.],[1.],[1.],[1]]
            col1, col2,col3,col4, col5, col6, col7, col8, col9, col10,col11,col12,col13,col14,col15,col16,col17 = tf.decode_csv(csv_row, record_defaults=record_defaults)
            features = tf.pack([col1,col2,col3,col4, col5,col6,col7, col8,col9, col10,col11,col12, col13,col14,col15,col16])
            col_t=tf.pack(col17)
        
        min_after_dequeue = 10000000
        capacity = min_after_dequeue +  define_net_resp.FEATURES * 1

        example_batch, label_batch_test = tf.train.batch([features,col14], batch_size=1, capacity=capacity,num_threads=1)
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for step_a in xrange(batch_size):
                   example, label = sess.run([example_batch, label_batch_test])
                   if  step_a==0:
                       array_data=np.concatenate((np.squeeze(example),label.astype(int)),axis=0)
                       a_r=np.zeros([1,array_data.shape[0]])
                   else:
                       array_data=np.concatenate((np.squeeze(example),label.astype(int)),axis=0)
                   if step_a!=int(sys.argv[5]):
                       a_r=np.concatenate((a_r,array_data.reshape(1,array_data.shape[0])), axis=0)
                   else:
                       np.savetxt(str(sys.argv[6]),array_data.reshape(1,array_data.shape[0]), delimiter=",",fmt="%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i")
            np.savetxt(str(sys.argv[3]),a_r, delimiter=",",fmt="%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%i")
            coord.request_stop()
            coord.join(threads)


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
