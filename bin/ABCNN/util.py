# coding: utf-8
import sys
import numpy as np
import numpy
import six
from collections import namedtuple
import random

import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, variable
import chainer.functions as F
import chainer.links as L


# basically this is same as the one on chainer's repo.
# I added padding option (padding=0) to be always true
def concat_examples(batch, x1s_len, x2s_len, device=None, padding=0):
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
                [example[i] for example in batch], padding[i], min_length)))

        return tuple(result)
    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            if key == "x1s":
                result[key] = to_device(_concat_xs(
                    [example[key] for example in batch], padding[key], x1s_len))
            elif key == "x2s":
                result[key] = to_device(_concat_xs(
                    [example[key] for example in batch], padding[key], x2s_len))
            else:
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
    # shape = min_length
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


def _concat_xs(arrays, padding, min_length):
    if padding is not None:
        return _concat_xs_with_padding(arrays, padding, min_length)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def _concat_xs_with_padding(arrays, padding, min_length):
    # shape = numpy.array(arrays[0].shape, dtype=int)
    shape = min_length
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
    norm_x = F.normalize(F.squeeze(x, axis=(1, 2)))
    norm_y = F.normalize(F.squeeze(y, axis=(1, 2)))
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
    name = 'SelectiveWeightDecay'

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


def compute_map_mrr(label_scores):
    """
    compute map and mrr
    argument is: numpy array with true label and predicted score
    """
    ap_list = []
    rr_list = []
    for label_score in label_scores:
        # sort (label, score) array, following score from the model
        sort_order = label_score[:, 1].argsort()[::-1]
        sorted_labels = label_score[sort_order][:, 0]  # split
        sorted_scores = label_score[sort_order][:, 1]  # split

        # compute map
        precision = 0
        correct_label = 0
        for n, (score, true) in enumerate(zip(sorted_scores, sorted_labels), start=1):
            if true == 1:
                correct_label += 1
                precision += (correct_label * 1.0 / n)
        ap = precision / correct_label
        ap_list.append(ap)

        # compute mrr
        ranks = [n for n, array in enumerate(
            label_score[sort_order], start=1) if int(array[0]) == 1]
        rr = (1.0 / ranks[0]) if ranks else 0.0
        rr_list.append(rr)

    Stats = namedtuple("Stats", ["map", "mrr"])
    return Stats(map=np.mean(ap_list), mrr=np.mean(rr_list))


def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)


def create_conv_param(output_channel, input_channel, embedding, filter_width):
    rng = np.random.RandomState(23455)
    filter_shape = [output_channel, input_channel, filter_width, embedding]
    fan_in = numpy.prod(filter_shape[1:])
    fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
    # initialize weights with random weights
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W = np.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
        dtype=np.float32)

    # the bias is a 1D tensor -- one bias per output feature map
    b=np.zeros((filter_shape[0],), dtype=np.float32)
    return W, b
