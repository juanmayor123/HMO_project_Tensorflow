import tensorflow as tf
import copy

epochs = 5
filename_seed = 4
capacity = 32
batch_size = 255

import string
import random

def make_random_string():
  return ''.join(random.choice(string.ascii_uppercase) for i in range(12))

input_size = 1024

random_strings = [make_random_string() for _ in range(input_size)]
random_strings2 = copy.deepcopy(random_strings)

if not (random_strings == random_strings2):
  raise BaseException('Input was not in correct order!')

def preprocess(x, whiten=False):
  return x

with tf.device('/cpu:0'):
  random_strings_queue = tf.train.string_input_producer(random_strings,
                                                        num_epochs=epochs, capacity=capacity,
                                                        shuffle=False)

  random_strings_queue2 = tf.train.string_input_producer(random_strings2,
                                                         num_epochs=epochs, capacity=capacity,
                                                         shuffle=False)

  batched = tf.train.batch([random_strings_queue.dequeue(), random_strings_queue2.dequeue()],
                            batch_size=batch_size, num_threads=16,
                            capacity=batch_size * 16)
init = tf.initialize_all_variables()

sess = tf.Session(config=tf.ConfigProto())
coord = tf.train.Coordinator()
sess.run(init)

threads = tf.train.start_queue_runners(sess=sess, coord=coord)
step = 0

try:
  while not coord.should_stop():
    batch_rs_1, batch_rs_2 = sess.run(batched)
    if not all(batch_rs_1 == batch_rs_2):
      print "\n".join(batch_rs_1)
      print
      print "\n".join(batch_rs_2)
      print step
      break
    step += 1

except tf.errors.OutOfRangeError:
  print 'Done training -- epoch limit reached'
finally:
  # When done, ask the threads to stop.
  coord.request_stop()
