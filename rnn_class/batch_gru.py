# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import theano
import theano.tensor as T

from util import init_weight


class GRU:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxr = init_weight(Mi, Mo)
        Whr = init_weight(Mo, Mo)
        br  = np.zeros(Mo)
        Wxz = init_weight(Mi, Mo)
        Whz = init_weight(Mo, Mo)
        bz  = np.zeros(Mo)
        Wxh = init_weight(Mi, Mo)
        Whh = init_weight(Mo, Mo)
        bh  = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxr = theano.shared(Wxr)
        self.Whr = theano.shared(Whr)
        self.br  = theano.shared(br)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz  = theano.shared(bz)
        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.bh  = theano.shared(bh)
        self.h0  = theano.shared(h0)
        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, self.h0]

    def get_ht(self, xWxr_t, xWxz_t, xWxh_t, h_t1):
      r = T.nnet.sigmoid(xWxr_t + h_t1.dot(self.Whr) + self.br)
      z = T.nnet.sigmoid(xWxz_t + h_t1.dot(self.Whz) + self.bz)
      hhat = self.f(xWxh_t + (r * h_t1).dot(self.Whh) + self.bh)
      h = (1 - z) * h_t1 + z * hhat
      return h

    def recurrence(self, xWxr_t, xWxz_t, xWxh_t, is_start, h_t1, h0):
        h_t = T.switch(
          T.eq(is_start, 1),
          self.get_ht(xWxr_t, xWxz_t, xWxh_t, h0),
          self.get_ht(xWxr_t, xWxz_t, xWxh_t, h_t1)
        )
        return h_t

    def output(self, Xflat, startPoints):
        # Xflat should be (NT, D)
        # calculate X after multiplying input weights
        XWxr = Xflat.dot(self.Wxr)
        XWxz = Xflat.dot(self.Wxz)
        XWxh = Xflat.dot(self.Wxh)

        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=[XWxr, XWxz, XWxh, startPoints],
            outputs_info=[self.h0],
            non_sequences=[self.h0],
            n_steps=Xflat.shape[0],
        )
        return h
