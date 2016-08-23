# -*- coding: utf-8 -*-

import flask
import predict
import json

app = flask.Flask(__name__)
# app.debug = True


@app.route('/predict', methods=['GET'])
def main():
    text = flask.request.args.get('text')
    res = predict.predict(text)
    return json.dumps(res, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    app.run()