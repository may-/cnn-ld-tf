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
    text = flask.request.args.get('text')
    config = {
        'data_dir': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'ted500'),
        'train_dir': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model', 'ted500'),
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


@app.route('/favicon.ico')
def favicon():
    return flask.send_from_directory(os.path.join(app.root_path, 'docs', 'img'),
                               'favicon.ico', mimetype='image/png')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    return response

if __name__ == '__main__':
    app.run()