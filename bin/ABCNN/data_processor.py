# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pickle
from collections import defaultdict
from itertools import groupby, islice


class DataProcessor(object):

    def __init__(self, data_path, vocab_path, test_run, max_length):
        self.train_data_path = os.path.join(data_path, "train.json")
        self.dev_data_path = os.path.join(data_path, "dev.json")
        self.test_data_path = os.path.join(data_path, "test.json")
        # conventional lexical features pkl used in [Yang+ 2015]
        self.id2features = pickle.load(open("../work/features.pkl", "rb"))
        self.test_run = test_run # if true, use tiny datasets for quick test
        self.max_length = max_length

        # Vocabulary for sentence pairs
        # word2vec vocabulary: vocab outside this will be considered as <unk>
        self.word2vec_vocab = {w.strip():1 for w in open(vocab_path, 'r')}
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.vocab["<pad>"]
        self.vocab["<unk>"]
        # 予測先のconnective tokens
        # self.connective = defaultdict(lambda: len(self.connective))

    def prepare_dataset(self):
        self.compute_max_length()
        # load train/dev/test data
        print("loading dataset...", end='', flush=True, file=sys.stderr)
        if self.test_run:
            print("...preparing tiny dataset for quick test...", end='', flush=True, file=sys.stderr)
        self.train_data, self.n_train = self.load_dataset("train")
        self.dev_data, self.n_dev = self.load_dataset("dev")
        self.test_data, self.n_test = self.load_dataset("test")
        print("done", flush=True, file=sys.stderr)

    def compute_max_length(self):
        end = 100 if self.test_run else none
        x1s_len = sorted([len(json.loads(l)['question']) for l in islice(open(self.train_data_path, 'r'), end)], reverse=True)[0]
        x2s_len = sorted([len(json.loads(l)['answer']) for l in islice(open(self.train_data_path, 'r'), end)], reverse=True)[0]

        self.max_x1s_len = x1s_len if x1s_len <= self.max_length else self.max_length
        self.max_x2s_len = x2s_len if x2s_len <= self.max_length else self.max_length

    def load_dataset(self, _type):
        if _type == "train":
            path = self.train_data_path
        elif _type == "dev":
            path = self.dev_data_path
        elif _type == "test":
            path = self.test_data_path

        dataset = []
        question_ids = []
        end = 100 if self.test_run else None
        with open(path, "r") as input_data:
            for line in islice(input_data, end):
                data = json.loads(line)
                y = np.array(data["label"], dtype=np.int32)
                x1s = np.array([self.vocab[token] if token in self.word2vec_vocab else self.vocab["<unk>"] for token in data["question"]][:self.max_x1s_len], dtype=np.int32)
                x2s = np.array([self.vocab[token] if token in self.word2vec_vocab else self.vocab["<unk>"] for token in data["answer"] ][:self.max_x2s_len], dtype=np.int32)  # truncate maximum 40 words
                x1s_len = np.array([len(x1s)], dtype=np.float32)
                x2s_len = np.array([len(x2s)], dtype=np.float32)
                wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wordcnt']], dtype=np.float32)
                wgt_wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wgt_wordcnt']], dtype=np.float32)
                question_ids.append(data['question_id'])
                sample = {
                    "x1s": x1s,
                    "x2s": x2s,
                    "wordcnt": wordcnt,
                    "wgt_wordcnt": wgt_wordcnt,
                    "x1s_len": x1s_len,
                    "x2s_len": x2s_len,
                    "y": y
                }
                dataset.append(sample)

        # Number of Question-Answer Pair for each question.
        # This is needed for validation, when calculating MRR and MAP
        qa_pairs = [len(list(section)) for _, section in groupby(question_ids)]
        return dataset, qa_pairs
