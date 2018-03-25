# -*- coding: utf-8 -*-

from datetime import datetime
import os
import tensorflow as tf
import numpy as np

import cnn
import util


FLAGS = tf.app.flags.FLAGS
this_dir = os.path.abspath(os.path.dirname(__file__))

tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data', 'ted500'), 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'model', 'ted500'), 'Where to read model')
tf.app.flags.DEFINE_boolean('train_data', False, 'Whether to evaluate on training data')


def evaluate(config):
    """ Build evaluation graph and run. """
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.global_variables())

        # read test files
        if FLAGS.train_data:
            loader = util.DataLoader(FLAGS.data_dir, 'train3.pkl', batch_size=FLAGS.batch_size)
        else:
            loader = util.DataLoader(FLAGS.data_dir, 'test3.pkl', batch_size=FLAGS.batch_size)
        print('Start evaluation, %d batches needed, with %d examples per batch.' % (loader.num_batch, FLAGS.batch_size))

        true_count = 0
        avg_loss = 0
        predictions = []

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            for _ in range(loader.num_batch):
                x_batch, y_batch = loader.next_batch()
                scores, loss_value, true_count_value = sess.run([m.scores, m.total_loss, m.true_count_op],
                                                                feed_dict={m.inputs: x_batch, m.labels: y_batch})

                predictions.extend(scores)
                true_count += true_count_value
                avg_loss += loss_value

            accuracy = float(true_count) / loader._num_examples
            avg_loss = float(avg_loss) / loader.num_batch
            print('%s: test_loss = %.6f, test_accuracy = %.3f' % (datetime.now(), avg_loss, accuracy))


def main(argv=None):
    FLAGS._parse_flags()
    config = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags3.pkl'))
    config['data_dir'] = FLAGS.data_dir
    config['train_dir'] = FLAGS.train_dir

    # predict
    evaluate(config)


if __name__ == '__main__':
    tf.app.run()
