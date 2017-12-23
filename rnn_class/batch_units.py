# https://deeplearningcourses.com/c/deep-learning-recurrent-neural-networks-in-python
# https://udemy.com/deep-learning-recurrent-neural-networks-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import theano
import theano.tensor as T

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) * np.sqrt(2.0 / Mi)


class SimpleRecurrentLayer:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxh = init_weight(Mi, Mo)
        Whh = init_weight(Mo, Mo)
        b   = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.b   = theano.shared(b)
        self.h0  = theano.shared(h0)
        self.params = [self.Wxh, self.Whh, self.b, self.h0]

    def get_ht(self, xWxh_t, h_t1):
      return self.f(xWxh_t + h_t1.dot(self.Whh) + self.b)

    def recurrence(self, xWxh_t, is_start, h_t1, h0):
        h_t = T.switch(
          T.eq(is_start, 1),
          self.get_ht(xWxh_t, h0),
          self.get_ht(xWxh_t, h_t1)
        )
        return h_t

    def output(self, Xflat, startPoints):
        # Xflat should be (NT, D)
        # calculate X after multiplying input weights
        XWxh = Xflat.dot(self.Wxh)

        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=[XWxh, startPoints],
            outputs_info=[self.h0],
            non_sequences=[self.h0],
            n_steps=Xflat.shape[0],
        )
        return h


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



class LSTM:
    def __init__(self, Mi, Mo, activation):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi  = np.zeros(Mo)
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf  = np.zeros(Mo)
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc  = np.zeros(Mo)
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo  = np.zeros(Mo)
        c0  = np.zeros(Mo)
        h0  = np.zeros(Mo)

        # theano vars
        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi  = theano.shared(bi)
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf  = theano.shared(bf)
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc  = theano.shared(bc)
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo  = theano.shared(bo)
        self.c0  = theano.shared(c0)
        self.h0  = theano.shared(h0)
        self.params = [
            self.Wxi,
            self.Whi,
            self.Wci,
            self.bi,
            self.Wxf,
            self.Whf,
            self.Wcf,
            self.bf,
            self.Wxc,
            self.Whc,
            self.bc,
            self.Wxo,
            self.Who,
            self.Wco,
            self.bo,
            self.c0,
            self.h0,
        ]

    def get_ht_ct(self, xWxi_t, xWxf_t, xWxc_t, xWxo_t, h_t1, c_t1):
        i_t = T.nnet.sigmoid(xWxi_t + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f_t = T.nnet.sigmoid(xWxf_t + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        c_t = f_t * c_t1 + i_t * T.tanh(xWxc_t + h_t1.dot(self.Whc) + self.bc)
        o_t = T.nnet.sigmoid(xWxo_t + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def recurrence(self, xWxi_t, xWxf_t, xWxc_t, xWxo_t, is_start, h_t1, c_t1, h0, c0):
        h_t_c_t = T.switch(
          T.eq(is_start, 1),
          self.get_ht_ct(xWxi_t, xWxf_t, xWxc_t, xWxo_t, h0, c0),
          self.get_ht_ct(xWxi_t, xWxf_t, xWxc_t, xWxo_t, h_t1, c_t1)
        )
        return h_t_c_t[0], h_t_c_t[1]

    def output(self, Xflat, startPoints):
        # Xflat should be (NT, D)
        # calculate X after multiplying input weights
        XWxi = Xflat.dot(self.Wxi)
        XWxf = Xflat.dot(self.Wxf)
        XWxc = Xflat.dot(self.Wxc)
        XWxo = Xflat.dot(self.Wxo)

        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=[XWxi, XWxf, XWxc, XWxo, startPoints],
            outputs_info=[self.h0, self.c0],
            non_sequences=[self.h0, self.c0],
            n_steps=Xflat.shape[0],
        )
        return h
