# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np

import cnn
import util

FLAGS = tf.app.flags.FLAGS


def predict(raw_x):
    """ Build evaluation graph and run. """
    vocab = util.VocabLoader(os.path.join(os.getcwd(), 'data', 'ted500'))
    x = vocab.text2id(raw_x)
    FLAGS.emb_size = 300
    FLAGS.batch_size = 100
    FLAGS.num_kernel= 100
    FLAGS.min_window = 3
    FLAGS.max_window = 5
    FLAGS.vocab_size = 4090
    FLAGS.num_classes = 65
    FLAGS.sent_len = 259
    FLAGS.l2_reg = 0.0


    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(FLAGS, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            train_dir = os.path.join(os.getcwd(), 'model', 'ted500')
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            scores = sess.run(m.scores, feed_dict={m.inputs:[x]})
            y_pred = np.argmax(scores[0])
            y_pred_class = vocab.class_names[int(y_pred)]
            scores = [float(str(i)) for i in scores[0]]
            scores_class = dict(zip(vocab.class_names, scores))

            ret = {}
            ret['prediction'] = y_pred_class
            ret['scores'] = scores_class

            return ret

def main(argv=None):
    text = u"日本語のテスト。"
    result = predict(text)
    print 'prediction = %s' % result['prediction']
    print 'scores = %s' % str(result['scores'])

if __name__ == '__main__':
    tf.app.run()
    #main()

