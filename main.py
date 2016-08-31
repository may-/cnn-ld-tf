# -*- coding: utf-8 -*-


import flask
import predict
import util

import json
import os


app = flask.Flask(__name__)
app.debug = True


@app.route('/predict', methods=['GET'])
def main():
    this_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(this_dir, 'data', 'ted500')
    restore_params = util.load_from_dump(os.path.join(data_dir, 'preprocess.cPickle'))
    text = flask.request.args.get('text')
    config = {
        'data_dir': data_dir,
        'train_dir': os.path.join(this_dir, 'model', 'ted500'),
        'emb_size': 300,
        'batch_size': 100,
        'num_kernel': 100,
        'min_window': 3,
        'max_window': 5,
        'vocab_size': restore_params['vocab_size'],
        'num_classes': len(restore_params['class_names']),
        'sent_len': restore_params['max_sent_len'],
        'l2_reg': 0.0
    }
    res = {}
    result = predict.predict(text, config, raw_text=True)
    language_codes = util.load_language_codes()
    res['prediction'] = language_codes[result['prediction']]
    res['scores'] = {language_codes[k]: v for k, v in result['scores'].iteritems()}
    return json.dumps(res, ensure_ascii=False, indent=2)


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