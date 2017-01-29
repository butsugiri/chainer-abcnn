# -*- coding: utf-8 -*-
"""
Add some arguments to StandardUpdater
"""
import six
from chainer.training import StandardUpdater
import copy
import six
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import variable

class ABCNNUpdater(StandardUpdater):
    def __init__(self, iterator, optimizer, converter, x1s_len, x2s_len, device=None, loss_func=None):
        super().__init__(iterator, optimizer, converter, device, loss_func)
        self.x1s_len = x1s_len
        self.x2s_len = x2s_len

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch=batch, device=self.device, x1s_len=self.x1s_len, x2s_len=self.x2s_len)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            optimizer.update(loss_func, *in_vars)
        elif isinstance(in_arrays, dict):
            in_vars = {key: variable.Variable(x)
                       for key, x in six.iteritems(in_arrays)}
            optimizer.update(loss_func, **in_vars)
        else:
            in_var = variable.Variable(in_arrays)
            optimizer.update(loss_func, in_var)
