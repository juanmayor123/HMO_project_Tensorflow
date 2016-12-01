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

hidden1=10
hidden2=10
hidden3=10
learning_rate=0.1
max_steps=6000

file_tt=str(sys.argv[1])
l_ex=file_len(file_tt)
filename_tt = tf.train.string_input_producer([file_tt],shuffle=False)
reader_tt = tf.TextLineReader(skip_header_lines=0)
_, csv_row_tt = reader_tt.read(filename_tt)
col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t = tf.decode_csv(csv_row_tt, record_defaults=record_defaults)
features_tt = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t])

min_after_dequeue = 10000000
    capacity = min_after_dequeue +  define_net_resp.FEATURES * batch_size


images_batch,label_batch = tf.train.batch([features_tt,col14t-1],batch_size=batch_size,
                                                          capacity=capacity,num_threads=1)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()