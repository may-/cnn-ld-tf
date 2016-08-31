# -*- coding: utf-8 -*-



import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import flask
import predict
import active_learning
import util

import json
import os
import time

import numpy as np

app = flask.Flask(__name__)
app.debug = True


@app.route('/predict', methods=['GET'])
def main():
    this_dir = os.path.abspath(os.path.dirname(__file__))
    text = flask.request.args.get('text')
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
    res = {}
    result = predict.predict(text, config, raw_text=True)
    language_codes = util.load_language_codes()
    res['prediction'] = language_codes[result['prediction']]
    res['scores'] = {language_codes[k]: v for k, v in result['scores'].iteritems()}
    return json.dumps(res, ensure_ascii=False, indent=2)


@app.route('/active', methods=['POST'])
def active():
    this_dir = os.path.abspath(os.path.dirname(__file__))

    # train_dir
    request = flask.request.get_json(silent=True)
    train_dir = os.path.join(this_dir, 'train', request['timestamp'])
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    data_dir = os.path.join(this_dir, 'data', request['dataset_name'])

    use_pretrain = False
    if os.path.exists(os.path.join(data_dir, 'emb.npy')):
        use_pretrain = True

    # parameters
    if os.path.exists(os.path.join(train_dir, 'flags.cPickle')):
        config = util.load_from_dump(os.path.join(train_dir, 'flags.cPickle'))
        assert data_dir == config['data_dir']
    else:
        config = {
            'data_dir': data_dir,
            'train_dir': train_dir,
            'num_epoch': 1,
            'strategy': 'max_entropy',
            'interactive': False,
            'use_pretrain': use_pretrain,
            'init_lr': 0.01,
            'current_lr': 0.01, # == init_lr
            'lr_decay': 0.95,
            'tolerance_step': 100,
            'emb_size': 300,
            'num_kernel': 100,
            'min_window': 3,
            'max_window': 5,
            'l2_reg': 0.0,
            'dropout': 0.5,
            'optimizer': 'adagrad',
            'lowest_loss_value': float('inf'),
            'global_step': 0,
            'decay_step_counter': 0,
            'train_loss': [],
            'train_accuracy': [],
            'dev_loss': [],
            'dev_accuracy': []
        }
    # dataset specific parameters
    restore_params = util.load_from_dump(os.path.join(data_dir, 'preprocess.cPickle'))
    config['batch_size'] = int(request['batch_size'])
    config['num_train_examples'] = restore_params['train_size']
    config['pool_size'] = max(restore_params['train_size']*0.1, 100)
    config['vocab_size'] = restore_params['vocab_size']
    config['num_classes'] = len(restore_params['class_names'])
    config['sent_len'] = restore_params['max_sent_len']

    # train step parameters
    if os.path.exists(os.path.join(config['train_dir'], 'parameters.cPickle')):
        parameters = util.load_from_dump(os.path.join(config['train_dir'], 'parameters.cPickle'))
        config['current_lr'] = parameters['current_lr']
        config['global_step'] = parameters['global_step']
        config['train_loss'] = parameters['train_loss']
        config['train_accuracy'] = parameters['train_accuracy']
        config['dev_loss'] = parameters['dev_loss']
        config['dev_accuracy'] = parameters['dev_accuracy']
        config['decay_step_counter'] = parameters['decay_step_counter']
        config['lowest_loss_value'] = parameters['lowest_loss_value']
    else: # initial values
        config['current_lr'] = config['init_lr']
        config['global_step'] = 0
        config['decay_step_counter'] = 0
        config['train_loss'] = []
        config['train_accuracy'] = []
        config['dev_loss'] = []
        config['dev_accuracy'] = []
        config['lowest_loss_value'] = float("inf")

    # active learning parameters
    if os.path.exists(os.path.join(train_dir, 'query_history.cPickle')):
        query_history = util.load_from_dump(os.path.join(train_dir, 'query_history.cPickle'))
    else:
        query_history = {
            'query_time': [],
            'pool_flag': [],
            'next_idx': range(config['batch_size']), # seed elements
            'random_seed': 1234
        }

    # load data
    np.random.seed(1234)
    #np.random.seed(query_history['random_seed']) # ensure the same permutation
    permutation = np.random.permutation(config['num_train_examples'])
    train_loader = util.DataLoader(config['data_dir'], 'train.cPickle',
        batch_size=config['batch_size'], shuffle=True, permutation=permutation, load_raw=True)
    config['num_batch'] = train_loader.num_batch
    for i in query_history['pool_flag']:
        train_loader.set_pool_flag(i)
    dev_loader = util.DataLoader(config['data_dir'], 'test.cPickle', batch_size=config['batch_size'])

    if config['global_step'] == 0:
        start_time = time.time()
        next_idx, pred = train_loader.next_batch_idx_active(config)
        query_time = time.time() - start_time

        x_batch = train_loader._x[next_idx]
        y_batch = train_loader._y[next_idx]
        #print 'AL query time: %.6f' % query_time

    else:
        start_time = time.time()
        next_idx = query_history['next_idx']
        query_time = time.time() - start_time

        x_batch = train_loader._x[next_idx]
        oracle = request['oracle']
        if len(oracle) > 0:
            assert len(oracle) == config['batch_size']
            y_batch = [[0] * config['num_classes'] for _ in xrange(config['batch_size'])]
            for i, j in enumerate(oracle):
                if j < config['num_classes']:
                    y_batch[i][j] = 1
        elif len(query_history['next_idx']) == config['batch_size']:
            y_batch = train_loader._y[query_history['next_idx']]
        else:
            raise Exception, 'oracle length != batch size'


    # annotation accuracy
    #agreement = {sum([1 for a in v if k[1] == a]): len(v) for k, v in train_loader.agreement.iteritems()}
    #if sum(agreement.values()) > 0:
    #    anno_acc = (sum(agreement.keys()) / float(sum(agreement.values())) )
    #    print 'annotation agreement: %3' % anno_acc

    current_params = active_learning.train(config, x_batch, y_batch, dev_loader)

    # return results
    train_loss = sum(current_params['train_loss']) / float(len(current_params['train_loss']))
    train_accuracy = sum(current_params['train_accuracy']) / float(len(current_params['train_accuracy']) * config['batch_size'])
    res = {
        'class_names': train_loader.class_names,
        'total_step': train_loader.num_batch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'dev_loss': current_params['dev_loss'][-1],
        'dev_accuracy': current_params['dev_accuracy'][-1],
        'global_step': config['global_step'] + 1,
        'current_lr': config['current_lr'],
        'timestamp': request['timestamp'],
        'query_time': query_time
    }

    # load next batch
    if config['num_batch'] > config['global_step']:
        next_idx, next_pred_y = train_loader.next_batch_idx_active(config)
        print 'pool_flag:', train_loader.get_pool_flag()[-1*config['batch_size']:]
        next_raw_x = train_loader._raw_x[next_idx]
        next_y = train_loader._y[next_idx]

        # update query_history
        query_history['query_time'].append(query_time)
        query_history['pool_flag'] = train_loader.get_pool_flag()
        query_history['next_idx'] = next_idx
        util.dump_to_file(os.path.join(train_dir, 'query_history.cPickle'), query_history)


        res['next_y'] = list(np.argmax(next_y, axis=1))
        res['next_pred_y'] = next_pred_y
        res['next_raw_x'] = list(next_raw_x)


    return json.dumps(res, ensure_ascii=False, encoding='utf-8', indent=2)


@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'docs', 'img'),
                               'favicon.ico', mimetype='image/png')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

if __name__ == '__main__':
    app.run()