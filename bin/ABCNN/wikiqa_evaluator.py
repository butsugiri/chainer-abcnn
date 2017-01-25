# coding: utf-8
import numpy as np
import copy
from chainer import reporter
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions
import random
import chainer

def average_precision(y_true, y_score):
    """
    y_true: actual label for y
    y_score: predicted value from the model

    this func computes average precision
    """
    correct_label = 0
    ap = 0
    for n, (true, score) in enumerate(zip(y_true, y_score), start=1):
        if true == 1:
            correct_label += 1
            ap += (correct_label * 1.0 / n)
    return ap / correct_label

class WikiQAEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, device, converter):
        super(WikiQAEvaluator, self).__init__(
            iterator=iterator, target=target, device=device, converter=converter)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        # this is necessary for more-than-once-evaluation
        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        for batch in it:
            x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len, y = self.converter(batch)
            x1s = chainer.Variable(x1s)
            x2s = chainer.Variable(x2s)
            observation = {}
            with reporter.report_scope(observation):
                # We evaluate WikiQA by MAP and MRR
                y_score = target(x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len)

                # compute average precision
                # do not use sklearn implementation
                label_score = np.c_[y, y_score.data]
                sort_order = label_score[:,1].argsort()[::-1]  #sort (label, score) array, following score from the model
                sorted_labels = label_score[sort_order][:,0]  # split
                sorted_scores = label_score[sort_order][:,1]  # split
                ap = average_precision(y_score=sorted_scores, y_true=sorted_labels)
                reporter.report({'map': ap}, target)


                # compute mean reciprocal rank
                label_score = np.vstack([y_score.data, y])
                sorted_label_score = label_score[
                    ::, label_score[0, ].argsort()[::-1]]
                ranks = [n for n, array in enumerate(sorted_label_score.T, start=1) if int(array[1]) == 1]
                rr = (1.0 / ranks[0]) if ranks else 0.0
                reporter.report({'mrr': rr}, target)

                # calculate loss
                loss = F.sigmoid_cross_entropy(x=y_score, t=y)
                reporter.report({'loss': loss}, target)

            summary.add(observation)
        return summary.compute_mean()
