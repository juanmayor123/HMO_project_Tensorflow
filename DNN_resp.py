from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics


import os.path
import time
import pdb

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

file_train = "train_sub3.csv"
file_test= "test_sub3.csv"
batch_size=1
FEATURES=13

def run_training():

#training_set = datasets.load_csv(filename=file_train,
#target_dtype=np.int)
                                                           #test_set = datasets.load_csv(filename=file_test,
                                                           #target_dtype=np.int)

    filename_queue = tf.train.string_input_producer([file_train],num_epochs=1,shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row = reader.read(filename_queue)
    filename_queue_test = tf.train.string_input_producer([file_test],num_epochs=1,shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row_test = reader.read(filename_queue_test)
    record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14 = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13])
    col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t = tf.decode_csv(csv_row_test, record_defaults=record_defaults)
    features_test = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t])
    min_after_dequeue = 100000
    capacity = min_after_dequeue + FEATURES * batch_size

    images_batch,label_batch = tf.train.batch([features,col14-1],
                                          batch_size=batch_size,
                                          capacity=capacity,num_threads=1)

    classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=4,optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1))
    coln=tf.to_int64(label_batch);
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print('hola')
        for step in xrange(1):
             example, label = sess.run([images_batch, label_batch])
             classifier.fit(example,label)
             print('hola')

    coord.request_stop()
    coord.join(threads)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()