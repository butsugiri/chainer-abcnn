# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
from chainer import cuda
from collections import defaultdict


class DataProcessor(object):

    def __init__(self, data_path, test):
        self.train_data_path = os.path.join(data_path, "train")
        self.dev_data_path = os.path.join(data_path, "dev")
        self.test_data_path = os.path.join(data_path, "test")
        self.test = test # if true, provide tiny datasets for quick test

        # Arg1/Arg2のボキャブラリ
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.vocab["<pad>"]
        # 予測先のconnective tokens
        self.connective = defaultdict(lambda: len(self.connective))

    def prepare_dataset(self):
        # load train/dev/test data
        sys.stderr.write("loading dataset...")
        self.train_data = self.load_dataset("train")
        self.dev_data = self.load_dataset("dev")
        if self.test:
            sys.stderr.write("...preparing tiny dataset for quick test...")
            self.train_data = self.train_data[:100]
            self.dev_data = self.dev_data[:10]
            # self.test_data = self.test_data[:10]
        sys.stderr.write("done.\n")

    def load_dataset(self, _type):
        if _type == "train":
            path = self.train_data_path
        elif _type == "dev":
            path = self.dev_data_path
        elif _type == "test":
            path = self.test_data_path

        dataset = []
        with open(path, "r") as input_data:
            for line in input_data:
                target, arg1, arg2 = line.strip().split("\t")
                y = np.array(self.connective[target], dtype=np.int32)

                arg1_tokens = arg1.strip().split(" ")
                x1s = np.array([self.vocab[token] for token in arg1_tokens], dtype=np.int32)
                arg2_tokens = arg2.strip().split(" ")
                x2s = np.array([self.vocab[token] for token in arg2_tokens], dtype=np.int32)

                dataset.append((x1s, x2s, y))
        return dataset
