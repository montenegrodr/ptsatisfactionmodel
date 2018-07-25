import argparse

from flask import Flask, request, jsonify
from flask.views import View
from ptsatmodel.classifiers import Classifiers

app = Flask(__name__)


class Model(View):

    methods = ['POST']

    def __init__(self, model):
        self.model = model

    def dispatch_request(self):
        text = request.form['text']
        predicted = self.model.predict([text])

        return jsonify(predicted[0])


def main(args):
    clf = Classifiers.factory(args.model)
    model = clf(args.storage)
    app.add_url_rule('/', view_func=Model.as_view('/', model))
    app.run('0.0.0.0', 8000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--storage')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
