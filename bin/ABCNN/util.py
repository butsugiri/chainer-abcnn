# coding: utf-8
import sys
import numpy as np
import numpy
import six

import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, variable
import chainer.functions as F
import chainer.links as L


# basically this is same as the one on chainer's repo.
# I added padding option (padding=0) to be always true
def concat_examples(batch, device=None, padding=0):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(_concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)
    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(_concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result


def _concat_arrays(arrays, padding):
    if padding is not None:
        return _concat_arrays_with_padding(arrays, padding)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def _concat_arrays_with_padding(arrays, padding):
    shape = numpy.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    shape = tuple(numpy.insert(shape, 0, len(arrays)))

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        result = xp.full(shape, padding, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src
    return result

def cos_sim(x, y):
    """
    Variableを2つ受け取ってcosine類似度を返す関数
    Chainerにはない
    """
    norm_x = F.normalize(F.squeeze(x, axis=(2,3)))
    norm_y = F.normalize(F.squeeze(y, axis=(2,3)))
    return F.batch_matmul(norm_x, norm_y, transa=True)

def debug_print(v):
    """
    print out chainer variable
    """
    try:
        assert isinstance(v, variable.Variable)
    except:
        raise AssertionError
    else:
        print(v.data)
        print(v.shape)

class SelectiveWeightDecay(object):
    name = 'WeightDecay'

    def __init__(self, rate, decay_params):
        self.rate = rate
        self.decay_params = decay_params

    def kernel(self):
        return cuda.elementwise(
            'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')

    def __call__(self, opt):
        rate = self.rate
        for name, param in opt.target.namedparams():
            if name in self.decay_params:
                p, g = param.data, param.grad
                with cuda.get_device(p) as dev:
                    if int(dev) == -1:
                        g += rate * p
                    else:
                        self.kernel()(p, rate, g)
