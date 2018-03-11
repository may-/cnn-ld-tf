# -*- coding: utf-8 -*-

from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import cnn
import util


FLAGS = tf.app.flags.FLAGS

# train parameters
this_dir = os.path.abspath(os.path.dirname(__file__))
tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data', 'ted500'), 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'train'), 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('num_epoch', 50, 'Number of epochs to run')
tf.app.flags.DEFINE_boolean('use_pretrain', False, 'Use pretrained embeddings or not')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use. One of "sgd", "adagrad", "adadelta" and "adam"')
tf.app.flags.DEFINE_float('init_lr', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'LR decay rate')
tf.app.flags.DEFINE_integer('tolerance_step', 500, 'Decay the lr after loss remains unchanged for this number of steps')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate. 0 is no dropout.')

# logging
tf.app.flags.DEFINE_integer('log_step', 10, 'Display log to stdout after this step')
tf.app.flags.DEFINE_integer('summary_step', 200, 'Write summary after this step')
tf.app.flags.DEFINE_integer('checkpoint_step', 200, 'Save model after this step')


def train():
    # train_dir
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, timestamp))

    # save flags
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    FLAGS._parse_flags()
    config = dict(FLAGS.__flags.items())
    util.dump_to_file(os.path.join(out_dir, 'flags3.pkl'), config)
    print("Parameters:")
    for k, v in config.iteritems():
        print('%20s %r' % (k, v))

    # load data
    print("Preparing train data ...")
    train_loader = util.DataLoader(FLAGS.data_dir, 'train3.pkl', batch_size=FLAGS.batch_size)
    print("Preparing test data ...")
    dev_loader = util.DataLoader(FLAGS.data_dir, 'test3.pkl', batch_size=FLAGS.batch_size)
    max_steps = train_loader.num_batch * FLAGS.num_epoch
    config['num_classes'] = train_loader.num_classes
    config['sent_len'] = train_loader.sent_len

    with tf.Graph().as_default():
        with tf.variable_scope('cnn', reuse=None):
            m = cnn.Model(config, is_train=True)
        with tf.variable_scope('cnn', reuse=True):
            mtest = cnn.Model(config, is_train=False)

        # checkpoint
        saver = tf.train.Saver(tf.global_variables())
        save_path = os.path.join(out_dir, 'model.ckpt')
        summary_op = tf.summary.merge_all()

        # session
        sess = tf.Session()

        # summary writer
        proj_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding = proj_config.embeddings.add()
        embedding.tensor_name = m.W_emb.name
        embedding.metadata_path = os.path.join(FLAGS.data_dir, 'metadata.tsv')
        summary_dir = os.path.join(out_dir, "summaries")
        summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, proj_config)

        sess.run(tf.global_variables_initializer())

        # assign pretrained embeddings
        if FLAGS.use_pretrain:
            print("Use pretrained embeddings to initialize model ...")
            emb_file = os.path.join(FLAGS.data_dir, 'emb.txt')
            vocab_file = os.path.join(FLAGS.data_dir, 'vocab.txt')
            pretrained_embedding = util.load_embedding(emb_file, vocab_file, FLAGS.vocab_size)
            m.assign_embedding(sess, pretrained_embedding)

        # initialize parameters
        current_lr = FLAGS.init_lr
        lowest_loss_value = float("inf")
        decay_step_counter = 0
        global_step = 0

        # evaluate on dev set
        def dev_step(mtest, sess, data_loader):
            dev_loss = 0.0
            dev_accuracy = 0.0
            for _ in range(data_loader.num_batch):
                x_batch_dev, y_batch_dev = data_loader.next_batch()
                dev_loss_value, dev_true_count = sess.run([mtest.total_loss, mtest.true_count_op],
                    feed_dict={mtest.inputs: x_batch_dev, mtest.labels: y_batch_dev})
                dev_loss += dev_loss_value
                dev_accuracy += dev_true_count
            dev_loss /= data_loader.num_batch
            dev_accuracy /= float(data_loader.num_batch * FLAGS.batch_size)
            data_loader.reset_pointer()
            return dev_loss, dev_accuracy

        # train loop
        print('\nStart training, %d batches needed, with %d examples per batch.' % (train_loader.num_batch, FLAGS.batch_size))
        for epoch in range(FLAGS.num_epoch):
            train_loss = []
            train_accuracy = []
            train_loader.reset_pointer()
            for _ in range(train_loader.num_batch):
                m.assign_lr(sess, current_lr)
                global_step += 1

                start_time = time.time()
                x_batch, y_batch = train_loader.next_batch()

                feed = {m.inputs: x_batch, m.labels: y_batch}
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_value, true_count = sess.run([m.train_op, m.total_loss, m.true_count_op],
                                                     feed_dict=feed, options=run_options, run_metadata=run_metadata)
                proc_duration = time.time() - start_time
                train_loss.append(loss_value)
                train_accuracy.append(true_count)

                assert not np.isnan(loss_value), "Model loss is NaN."

                if global_step % FLAGS.log_step == 0:
                    examples_per_sec = FLAGS.batch_size / proc_duration
                    accuracy = float(true_count) / FLAGS.batch_size
                    format_str = '%s: step %d/%d (epoch %d/%d), acc = %.2f, loss = %.2f ' + \
                                 '(%.1f examples/sec; %.3f sec/batch), lr: %.6f'
                    print(format_str % (datetime.now(), global_step, max_steps, epoch+1, FLAGS.num_epoch,)
                                        accuracy, loss_value, examples_per_sec, proc_duration, current_lr)

                # write summary
                if global_step % FLAGS.summary_step == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_run_metadata(run_metadata, 'step%04d' % global_step)
                    summary_writer.add_summary(summary_str, global_step)

                    # summary loss/accuracy
                    train_loss_mean = sum(train_loss) / float(len(train_loss))
                    train_accuracy_mean = sum(train_accuracy) / float(len(train_accuracy) * FLAGS.batch_size)
                    summary_writer.add_summary(_summary('train/loss', train_loss_mean), global_step=global_step)
                    summary_writer.add_summary(_summary('train/accuracy', train_accuracy_mean), global_step=global_step)

                    test_loss, test_accuracy = dev_step(mtest, sess, dev_loader)
                    summary_writer.add_summary(_summary('dev/loss', test_loss), global_step=global_step)
                    summary_writer.add_summary(_summary('dev/accuracy', test_accuracy), global_step=global_step)

                    print("\nStep %d: train_loss = %.6f, train_accuracy = %.3f" % (global_step, train_loss_mean, train_accuracy_mean))
                    print("Step %d:  test_loss = %.6f,  test_accuracy = %.3f\n" % (global_step, test_loss, test_accuracy))

                # decay learning rate if necessary
                if loss_value < lowest_loss_value:
                    lowest_loss_value = loss_value
                    decay_step_counter = 0
                else:
                    decay_step_counter += 1
                if decay_step_counter >= FLAGS.tolerance_step:
                    current_lr *= FLAGS.lr_decay
                    print('%s: step %d/%d (epoch %d/%d), Learning rate decays to %.5f' % \
                          (datetime.now(), global_step, max_steps, epoch+1, FLAGS.num_epoch, current_lr))
                    decay_step_counter = 0

                # stop learning if learning rate is too low
                if current_lr < 1e-5:
                    break

                # save checkpoint
                if global_step % FLAGS.checkpoint_step == 0:
                    saver.save(sess, save_path, global_step=global_step)
        saver.save(sess, save_path, global_step=global_step)


def _summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])


def main(argv=None):
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
