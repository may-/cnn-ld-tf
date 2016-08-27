# -*- coding: utf-8 -*-

from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import cnn
import util

FLAGS = tf.app.flags.FLAGS
this_dir = os.path.abspath(os.path.dirname(__file__))

tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data', 'ted500'), 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'model', 'ted500'), 'Where to read model')
tf.app.flags.DEFINE_boolean('train_data', False, 'To evaluate on training data')

def evaluate():
    """ Build evaluation graph and run. """
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(FLAGS, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        # read test files
        if FLAGS.train_data:
            loader = util.DataLoader(FLAGS.data_dir, 'train.cPickle', batch_size=FLAGS.batch_size)
        else:
            loader = util.DataLoader(FLAGS.data_dir, 'test.cPickle', batch_size=FLAGS.batch_size)
        print 'Start evaluation, %d batches needed, with %d examples per batch.' % (loader.num_batch, FLAGS.batch_size)

        true_count = 0
        avg_loss = 0
        predictions = []

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")


            for _ in xrange(loader.num_batch):
                x_batch, y_batch = loader.next_batch()
                scores, loss_value, true_count_value = sess.run([m.scores, m.total_loss, m.true_count_op],
                                                        feed_dict={m.inputs: x_batch, m.labels: y_batch})

                predictions.extend(scores)
                true_count += true_count_value
                avg_loss += loss_value

            accuracy = float(true_count) / loader._num_examples
            avg_loss = float(avg_loss) / loader.num_batch
            print '%s: test_loss = %.6f, test_accuracy = %.3f' % (datetime.now(), avg_loss, accuracy)

            return predictions

def main(argv=None):
    predictions = evaluate()
    scores = np.vstack(tuple(predictions))
    util.dump_to_file(os.path.join(FLAGS.data_dir, 'eval'), scores)

if __name__ == '__main__':
    tf.app.run()

