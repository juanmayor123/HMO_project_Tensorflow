from __future__ import print_function
import tensorflow as tf
from collections import namedtuple
import glob
import os
#from iris import log

NUM_HIDDEN = 2
NUM_FEATURES = 13
NUM_LABELS = 4
cwd = os.getcwd()
#_logger = log.get_logger()

Topology = namedtuple(
                      "Topology",
                      "x y t w_hidden b_hidden hidden w_out b_out")


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def input_pipeline(batch_size,example,label):
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity,min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

def do_eval(sess,correct,data_test,labels_test,x,y_):
     true_cont=0
     for step in range(120):
        ex,lab = sess.run([data_test, labels_test])
        ll=tf.one_hot(lab-1,4,1,0)
        print(sess.run(correct, feed_dict={x: [ex], y_: [ll.eval()]}))
        #print(true_cont)
#print(true_cont)

filename = "train_sub1.csv"
file_name_test= "test_sub1.csv"
# setup text reader
file_length = file_len(filename)
filename_queue = tf.train.string_input_producer([filename])
reader = tf.TextLineReader(skip_header_lines=0)
_, csv_row = reader.read(filename_queue)
filename_queue_test = tf.train.string_input_producer([file_name_test])
reader = tf.TextLineReader(skip_header_lines=0)
_, csv_row_test = reader.read(filename_queue_test)


# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1.], [1.], [1.], [1.], [1.],[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13, col14 = tf.decode_csv(csv_row, record_defaults=record_defaults)
features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,  col11,  col12,  col13])
col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t, col14t = tf.decode_csv(csv_row_test, record_defaults=record_defaults)
features_test = tf.pack([col1t, col2t, col3t, col4t, col5t, col6t, col7t, col8t, col9t, col10t,  col11t,  col12t,  col13t])

x = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])
W = tf.Variable(tf.zeros([NUM_FEATURES, 4]))
b = tf.Variable(tf.zeros([4]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, tf.float32))


print("loading, " + str(file_length) + " line(s)\n")
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #lab=tf.one_hot(tf.to_int64(col14)-1,NUM_LABELS,1,0)
    for step in range(file_length):
    # retrieve a single instance
        example,label = sess.run([features, col14])
        ll=tf.one_hot(label-1,4,1,0)
        print(example,label,ll.eval())
        sess.run(train_step,feed_dict={x: [example], y_: [ll.eval()]})
        if (step + 1) % 100 == 0 or (step + 1) == file_length:
            checkpoint_file = os.path.join(cwd,'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
    
    do_eval(sess,tf_accuracy,features_test,col14t,x,y_)
    coord.request_stop()
    coord.join(threads)
    print("\ndone loading")

#[train_feat,lab_train]=input_pipeline(10,features,lab)
#train(np.array([ i[1::] for i in train_feat]),lab,100)