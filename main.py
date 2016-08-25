# -*- coding: utf-8 -*-

import flask, send_from_directory
import predict
import json
import os

app = flask.Flask(__name__)
app.debug = True


@app.route('/predict', methods=['GET'])
def main():
    text = flask.request.args.get('text')
    res = predict.predict(text.strip())
    return json.dumps(res, ensure_ascii=False, indent=2)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'docs', 'img'),
                               'favicon.ico', mimetype='image/png')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    return response

if __name__ == '__main__':
    app.run()