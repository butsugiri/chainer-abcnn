# -*- coding: utf-8 -*-
"""
custom classifier for dict as argument of __call__
"""
import sys
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter

class Classifier(link.Chain):
    compute_accuracy = True

    def __init__(self, predictor, lossfun=softmax_cross_entropy.softmax_cross_entropy, accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None


    def __call__(self, **in_vars):
        # print(in_vars)
        assert len(in_vars) >= 2
        x = {k:v for k,v in in_vars.items() if k != 'y'}
        t = in_vars['y']
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(**x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
