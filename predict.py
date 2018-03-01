# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np

import cnn
import util


def predict(x, config, raw_text=True):
    """ Build evaluation graph and run. """
    if raw_text:
        vocab = util.VocabLoader(config['data_dir'])
        x = vocab.text2id(x)
        class_names = vocab.class_names

        x_input = np.array([x])
    else:
        x_input = x

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            train_dir = config['train_dir']
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

    ret = dict()
    ret['prediction'] = y_pred
    ret['scores'] = scores
    return ret


def main(argv=None):
    FLAGS = tf.app.flags.FLAGS
    this_dir = os.path.abspath(os.path.dirname(__file__))
    tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data', 'ted500'), 'Directory of the data')
    tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'model', 'ted500'), 'Where to read model')
    FLAGS._parse_flags()

    text = u"日本語のテスト。"
    config = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags3.pkl'))
    config['data_dir'] = FLAGS.data_dir
    config['train_dir'] = FLAGS.train_dir

    # predict
    result = predict(text, config, raw_text=True)
    language_codes = util.load_language_codes()
    print('prediction = %s' % language_codes[result['prediction']])
    print('scores = %s' % str({language_codes[k]: v for k, v in result['scores'].iteritems()}))


if __name__ == '__main__':
    tf.app.run()
