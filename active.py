# -*- coding: utf-8 -*-

from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import cnn
import util


FLAGS = tf.app.flags.FLAGS


def train(config, x_batch, y_batch, dev_loader):
    FLAGS.data_dir = config['data_dir']
    FLAGS.train_dir = config['train_dir']

    # active learning parameters
    #FLAGS.active= True
    #FLAGS.pool_size = config['pool_size']
    #FLAGS.strategy = config['strategy']
    #FLAGS.checkpoint_step = config['checkpoint_step']

    # train parameters
    FLAGS.batch_size = config['batch_size']
    FLAGS.num_classes = config['num_classes'] #train_loader.num_classes
    FLAGS.sent_len = config['sent_len'] #train_loader.sent_len
    #FLAGS.init_lr = config['init_lr']
    #FLAGS.lr_decay = config['lr_decay']
    #FLAGS.tolerance_step = config['tolerance_step']

    # model parameters
    FLAGS.dropout = config['dropout']
    FLAGS.optimizer = config['optimizer']
    FLAGS.emb_size = config['emb_size']
    FLAGS.num_kernel= config['num_kernel']
    FLAGS.min_window = config['min_window']
    FLAGS.max_window = config['max_window']
    FLAGS.vocab_size = config['vocab_size']
    FLAGS.l2_reg = config['l2_reg']



    with tf.Graph().as_default():
        with tf.variable_scope('cnn', reuse=None):
            m = cnn.Model(FLAGS, is_train=True)
        with tf.variable_scope('cnn', reuse=True):
            mtest = cnn.Model(FLAGS, is_train=False)

        saver = tf.train.Saver(tf.all_variables())
        save_path = os.path.join(config['train_dir'], 'model.ckpt')
        #summary_op = tf.merge_all_summaries()

        # initialize parameters
        max_steps = config['num_batch']
        current_lr = config['init_lr']
        lowest_loss_value = float("inf")
        decay_step_counter = 0
        global_step = 0
        train_loss = []
        train_accuracy = []
        dev_loss = []
        dev_accuracy = []

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            #summary_writer = tf.train.SummaryWriter(summary_dir, graph_def=sess.graph_def)
            #summary_dir = os.path.join(config['train_dir'], "summaries")
            #summary_writer = tf.train.SummaryWriter(summary_dir, graph=sess.graph)
            sess.run(tf.initialize_all_variables())

            # restore parameters
            if os.path.exists(os.path.join(config['train_dir'], 'parameters')):
                parameters = util.load_from_dump(os.path.join(config['train_dir'], 'parameters'))
                current_lr = parameters['current_lr']
                global_step = parameters['global_step']
                train_loss = parameters['train_loss']
                train_accuracy = parameters['train_accuracy']
                dev_loss = parameters['dev_loss']
                dev_accuracy = parameters['dev_accuracy']
                decay_step_counter = parameters['decay_step_counter']
                lowest_loss_value = parameters['lowest_loss_value']

            # restore checkpoint
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # assign pretrained embeddings
            if config['use_pretrain'] and global_step == 0:
                print "Use pretrained embeddings to initialize model ..."
                pretrained_embedding = np.load(os.path.join(config['data_dir'], 'emb.npy'))
                m.assign_embedding(sess, pretrained_embedding)

            # evaluation on dev set
            def dev_step(mtest, sess, data_loader):
                dev_loss = 0.0
                dev_accuracy = 0.0
                data_loader.reset_pointer()
                for _ in xrange(data_loader.num_batch):
                    x_batch, y_batch = data_loader.next_batch()
                    loss_value, true_count = sess.run([mtest.total_loss, mtest.true_count_op],
                                                      feed_dict={mtest.inputs: x_batch, mtest.labels: y_batch})
                    dev_loss += loss_value
                    dev_accuracy += true_count
                dev_loss /= data_loader.num_batch
                dev_accuracy /= float(data_loader.num_batch * config['batch_size'])
                data_loader.reset_pointer()
                return (dev_loss, dev_accuracy)

            # no loop here
            #for _ in xrange(train_loader.num_batch):
            m.assign_lr(sess, current_lr)
            global_step += 1

            start_time = time.time()
            feed = {m.inputs: x_batch, m.labels: y_batch}
            _, loss_value, true_count = sess.run([m.train_op, m.total_loss, m.true_count_op], feed_dict=feed)
            duration = time.time() - start_time
            train_loss.append(loss_value)
            train_accuracy.append(true_count)

            assert not np.isnan(loss_value), "Model loss is NaN."

            accuracy = float(true_count) / config['batch_size']
            # batch log
            print '\n%s: Step %d/%d, acc = %.2f, loss = %.2f, (%.3f sec/batch), lr = %.6f' % \
                  (datetime.now(), global_step, max_steps, accuracy, loss_value, duration, current_lr)


            # decay learning rate if necessary
            if loss_value < lowest_loss_value:
                lowest_loss_value = loss_value
                decay_step_counter = 0
            else:
                decay_step_counter += 1
            if decay_step_counter >= config['tolerance_step']:
                current_lr *= config['lr_decay']
                print '%s: Step %d/%d LR decays to %.5f' % (datetime.now(), global_step, max_steps, current_lr)
                decay_step_counter = 0

            # stop learning if learning rate is too low
            if current_lr < 1e-5: return


            # write summary
            #summary_str = sess.run(summary_op)
            #summary_writer.add_summary(summary_str, global_step)

            # train summary
            train_loss_mean = sum(train_loss) / float(len(train_loss))
            train_accuracy_mean = sum(train_accuracy) / float(len(train_accuracy) * config['batch_size'])
            #summary_writer.add_summary(_summary_for_scalar('eval/train_loss', train_loss_mean), global_step=global_step)
            #summary_writer.add_summary(_summary_for_scalar('eval/train_accuracy', train_accuracy_mean), global_step=global_step)

            # dev_summary
            dev_step_loss, dev_step_accuracy = dev_step(mtest, sess, dev_loader)
            #summary_writer.add_summary(_summary_for_scalar('dev/loss', dev_loss), global_step=global_step)
            #summary_writer.add_summary(_summary_for_scalar('dev/accuracy', dev_accuracy), global_step=global_step)

            print "train_loss = %.6f, train_accuracy = %.3f" % (train_loss_mean, train_accuracy_mean)
            print "  dev_loss = %.6f,   dev_accuracy = %.3f" % (dev_step_loss, dev_step_accuracy)
            dev_loss.append(dev_step_loss)
            dev_accuracy.append(dev_step_accuracy)

            # save checkpoint
            saver.save(sess, save_path, global_step=global_step)
            # remove old checkpoints
            if global_step > 3:
                os.remove(os.path.join(config['train_dir'], 'model.ckpt-%d' % (global_step - 3)))
                os.remove(os.path.join(config['train_dir'], 'model.ckpt-%d.meta' % (global_step - 3)))


            # save parameters
            obj = {
                'global_step': global_step,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'dev_loss': dev_loss,
                'dev_accuracy': dev_accuracy,
                'current_lr': current_lr,
                'decay_step_counter': decay_step_counter,
                'duration': duration,
                'lowest_loss_value': lowest_loss_value
            }
            util.dump_to_file(os.path.join(config['train_dir'], 'parameters'), obj)

    return

#def _summary_for_scalar(name, value):
#    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])

def main(argv=None):
    this_dir = os.path.abspath(os.path.dirname(__file__))
    # train_dir
    timestamp = str(int(time.time()))
    train_dir = os.path.join(this_dir, 'train', timestamp)
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MakeDirs(train_dir)

    # parameters
    config = {
        'data_dir': os.path.join(this_dir, 'data', 'mr'),
        'train_dir': train_dir,
        'batch_size': 100,
        'pool_size': 1000,
        'strategy': 'max_entropy',
        'interactive': False,
        'use_pretrain': True,
        'init_lr': 0.01,
        'lr_decay': 0.95,
        'tolerance_step': 100,
        'emb_size': 300,
        'num_kernel': 100,
        'min_window': 3,
        'max_window': 5,
        'vocab_size': 15000,
        'num_classes': 2,   # train_loader.num_classes
        'sent_len': 56,     # train_loader.sent_len
        'l2_reg': 0.0,
        'dropout': 0.5,
        'optimizer': 'adagrad',
    }

    # load data
    train_loader = util.DataLoader(config['data_dir'], 'train.cPickle', batch_size=config['batch_size'], load_raw=True)
    config['num_batch'] = train_loader.num_batch
    train_loader.reset_pointer()
    dev_loader = util.DataLoader(config['data_dir'], 'test.cPickle', batch_size=config['batch_size'])

    query_time = []

    # loop here
    for _ in xrange(train_loader.num_batch):
        start_time = time.time()
        next_idx = train_loader.next_batch_idx_active(config)
        x_batch = train_loader._x[next_idx]
        if config['interactive']:
            y_batch = train_loader.get_oracle(next_idx)
        else:
            y_batch = train_loader._y[next_idx]
        query_time.append(time.time() - start_time)
        #print 'AL query time: %.6f' % query_time[-1]

        # annotation accuracy
        #if config['interactive']:
        #   agreement = {sum([1 for a in v if k[1] == a]): len(v) for k, v in train_loader.agreement.iteritems()}
        #   if sum(agreement.values()) > 0:
        #       anno_acc = (sum(agreement.keys()) / float(sum(agreement.values())) )
        #       print 'annotation agreement: %3' % anno_acc

        train(config, x_batch, y_batch, dev_loader)

    obj = {'query_time': query_time, 'agreement': train_loader.agreement}
    util.dump_to_file(os.path.join(train_dir, 'query_history'), obj)


if __name__ == '__main__':
    tf.app.run()