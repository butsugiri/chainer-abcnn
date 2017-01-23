# coding: utf-8
import numpy as np
import copy
from sklearn.metrics import average_precision_score
from chainer import reporter
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions

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
            x1s, x2s, y = self.converter(batch)
            observation = {}
            with reporter.report_scope(observation):
                # We evaluate WikiQA by MAP and MRR
                y_score = target(x1s, x2s)

                # calculate average precision
                ap = average_precision_score(y_true=y, y_score=y_score.data)
                reporter.report({'average_precision': ap}, target)

                # calculate mean reciprocal rank
                label_score = np.vstack([y_score.data, y])
                sorted_label_score = label_score[
                    ::, label_score[0, ].argsort()[::-1]]
                ranks = [n for n, array in enumerate(sorted_label_score.T, start=1) if array[
                    0] > 0 and int(array[1]) == 1]
                rr = (1.0 / ranks[0]) if ranks else 0.0
                reporter.report({'reciprocal_rank': rr}, target)

                # calculate loss
                loss = F.sigmoid_cross_entropy(x=y_score, t=y)
                reporter.report({'loss': loss}, target)

            summary.add(observation)
        return summary.compute_mean()
