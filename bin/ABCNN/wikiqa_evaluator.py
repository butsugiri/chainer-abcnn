# coding: utf-8
import numpy as np
import copy
from chainer import reporter
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions
from sklearn.svm import LinearSVC
from collections import namedtuple
import random
import chainer

class WikiQAEvaluator(extensions.Evaluator):

    def __init__(self, iterator, target, device, converter):
        super(WikiQAEvaluator, self).__init__(
            iterator=iterator, target=target, device=device, converter=converter)

    def collect_prediction_for_train_data(self):
        """
        collect prediction scores from the model.
        this is needed for training SVM/LR
        """
        iterator = self._iterators['train']
        target = self._targets['main']
        it = copy.copy(iterator)

        train_X = []
        train_y = []
        for batch in it:
            x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len, y = self.converter(batch)
            y_score, similarity_score = target(x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len)
            x = np.concatenate([similarity_score.data, wordcnt, wgt_wordcnt], axis=1)
            train_X.append(x)
            train_y.append(y)

        train_X = np.concatenate(train_X, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        return train_X, train_y


    def evaluate(self):

        train_X, train_y = self.collect_prediction_for_train_data()
        model = LinearSVC()
        model.fit(X=train_X, y=train_y)

        iterator = self._iterators['dev']
        target = self._targets['main']
        # this is necessary for more-than-once-evaluation
        it = copy.copy(iterator)

        label_scores = []
        svm_label_scores = []
        summary = reporter.DictSummary()
        for n, batch in enumerate(it):
            observation = {}
            with reporter.report_scope(observation):
                x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len, y = self.converter(batch)
                y_score, similarity_score = target(x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len)

                # compute loss
                loss = F.sigmoid_cross_entropy(x=y_score, t=y).data
                reporter.report({'loss': loss}, target)

                # We evaluate WikiQA by MAP and MRR
                # for direct evaluation
                label_score = np.c_[y, y_score.data]
                label_scores.append(label_score)
                # for SVM/LR
                x = np.concatenate([similarity_score.data, wordcnt, wgt_wordcnt], axis=1)
                y_score = model.decision_function(x)
                svm_label_score = np.c_[y, y_score]
                svm_label_scores.append(svm_label_score)
            summary.add(observation)

        stats = compute_map_mrr(label_scores)
        svm_stats = compute_map_mrr(svm_label_scores)
        summary_dict = summary.compute_mean()
        summary_dict["validation/main/svm_map"] = svm_stats.map
        summary_dict["validation/main/svm_mrr"] = svm_stats.mrr
        summary_dict["validation/main/map"] = stats.map
        summary_dict["validation/main/mrr"] = stats.mrr
        return summary_dict


def compute_map_mrr(label_scores):
    """
    compute map and mrr
    argument is: numpy array with true label and predicted score
    """
    ap_list = []
    rr_list = []
    for label_score in label_scores:
        sort_order = label_score[:,1].argsort()[::-1]  #sort (label, score) array, following score from the model
        sorted_labels = label_score[sort_order][:,0]  # split
        sorted_scores = label_score[sort_order][:,1]  # split

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
        ranks = [n for n, array in enumerate(label_score[sort_order], start=1) if int(array[0]) == 1]
        rr = (1.0 / ranks[0]) if ranks else 0.0
        rr_list.append(rr)

    Stats = namedtuple("Stats", ["map", "mrr"])
    return Stats(map=np.mean(ap_list), mrr=np.mean(rr_list))
