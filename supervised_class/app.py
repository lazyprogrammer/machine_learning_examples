from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
import pickle
import numpy as np
import os
import json

import tornado.ioloop
import tornado.web

if not os.path.exists('mymodel.pkl'):
    exit("Can't run without the model!")

with open('mymodel.pkl', 'rb') as f:
    model = pickle.load(f)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, Tornado!")

class PredictionHandler(tornado.web.RequestHandler):
    # predict one sample at a time
    def post(self):
        # print "body:", self.request.body
        # print "arguments:", self.request.arguments
        # will look like this:
        # body: three=four&one=two
        # arguments: {'three': ['four'], 'one': ['two']}
        params = self.request.arguments
        x = np.array(list(map(float, params['input'])))
        y = model.predict([x])[0]
        self.write(json.dumps({'prediction': y.item()}))
        self.finish()

if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/predict", PredictionHandler),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
