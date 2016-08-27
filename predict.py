# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np

import cnn
import util

FLAGS = tf.app.flags.FLAGS

def predict(x, config, raw_text=True):
    """ Build evaluation graph and run. """
    if raw_text:
        vocab = util.VocabLoader(config['data_dir'])  # data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'ted500')
        x = vocab.text2id(x)
        class_names = vocab.class_names

        x_input = np.array([x])
    else:
        x_input = x

    # model parameters
    FLAGS.emb_size = config['emb_size']
    FLAGS.batch_size = config['batch_size']
    FLAGS.num_kernel= config['num_kernel']
    FLAGS.min_window = config['min_window']
    FLAGS.max_window = config['max_window']
    FLAGS.vocab_size = config['vocab_size']
    FLAGS.num_classes = config['num_classes']
    FLAGS.sent_len = config['sent_len']
    FLAGS.l2_reg = config['l2_reg']


    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(FLAGS, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            train_dir = config['train_dir'] # train_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'ted500')
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            scores = sess.run(m.scores, feed_dict={m.inputs:x_input})

    if raw_text:
        scores = [float(str(i)) for i in scores[0]]
        y_pred = class_names[int(np.argmax(scores))]
        scores = dict(zip(class_names, scores))
    else:
        y_pred = np.argmax(scores, axis=1)


    ret = {}
    ret['prediction'] = y_pred
    ret['scores'] = scores
    return ret

def main(argv=None):
    text = u"日本語のテスト。"
    this_dir = os.path.abspath(os.path.dirname(__file__))
    config = {
        'data_dir': os.path.join(this_dir, 'data', 'ted500'),
        'train_dir': os.path.join(this_dir, 'model', 'ted500'),
        'emb_size': 300,
        'batch_size': 100,
        'num_kernel': 100,
        'min_window': 3,
        'max_window': 5,
        'vocab_size': 4090,
        'num_classes': 65,
        'sent_len': 259,
        'l2_reg': 0.0
    }
    result = predict(text, config, raw_text=True)
    language_codes = util.load_language_codes()
    print 'prediction = %s' % language_codes[result['prediction']]
    print 'scores = %s' % str({language_codes[k]: v for k, v in result['scores'].iteritems()})

if __name__ == '__main__':
    tf.app.run()
    #main()

