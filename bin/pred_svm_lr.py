# -*- coding: utf-8 -*-
"""
use output from predict.py
"""
from itertools import groupby
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from ABCNN import compute_map_mrr
import numpy as np

train_data_path = "../data/WikiQACorpus/train.prediction"
dev_data_path = "../data/WikiQACorpus/dev.prediction"
test_data_path = "../data/WikiQACorpus/test.prediction"

def load_data(path):
    Xs = []
    ys = []
    with open(path, 'r') as fi:
        for is_empty, section in groupby(fi, lambda x: x.strip() == ''):
            if not is_empty:
                section = list(section)
                y = np.array([x.strip().split("\t")[1] for x in section], dtype=np.int32)
                X = np.array([x.strip().split("\t")[2::] for x in section], dtype=np.float32)
                ys.append(y)
                Xs.append(X)
    return Xs, ys

def main():
    train_X, train_y = load_data(train_data_path)
    dev_X, dev_y = load_data(dev_data_path)
    test_X, test_y = load_data(test_data_path)

    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)

    #### SVM ####
    # train SVM
    model = LinearSVC()
    model.fit(X=train_X, y=train_y)

    # compute map and mrr for dev and test
    svm_label_scores = []
    for xi, yi in zip(dev_X, dev_y):
        y_score = model.decision_function(xi)
        svm_label_score = np.c_[yi, y_score]
        svm_label_scores.append(svm_label_score)
    stats = compute_map_mrr(svm_label_scores)
    print("SVM Dev\nMAP:{}\tMRR:{}\n".format(stats.map, stats.mrr))

    svm_label_scores = []
    for xi, yi in zip(test_X, test_y):
        y_score = model.decision_function(xi)
        svm_label_score = np.c_[yi, y_score]
        svm_label_scores.append(svm_label_score)
    stats = compute_map_mrr(svm_label_scores)
    print("SVM Test\nMAP:{}\tMRR:{}\n".format(stats.map, stats.mrr))


    #### Logistic Regression ####
    # train LR
    model = LogisticRegression()
    model.fit(X=train_X, y=train_y)

    # compute map and mrr for dev and test
    lr_label_scores = []
    for xi, yi in zip(dev_X, dev_y):
        y_score = model.decision_function(xi)
        lr_label_score = np.c_[yi, y_score]
        lr_label_scores.append(lr_label_score)
    stats = compute_map_mrr(lr_label_scores)
    print("LR Dev\nMAP:{}\tMRR:{}\n".format(stats.map, stats.mrr))

    lr_label_scores = []
    for xi, yi in zip(test_X, test_y):
        y_score = model.decision_function(xi)
        lr_label_score = np.c_[yi, y_score]
        lr_label_scores.append(lr_label_score)
    stats = compute_map_mrr(lr_label_scores)
    print("LR Test\nMAP:{}\tMRR:{}\n".format(stats.map, stats.mrr))


if __name__ == "__main__":
    main()
