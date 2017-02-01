# -*- coding: utf-8 -*-
"""
compare word2vec vocabulary and dataset vocabulary
"""
import sys
import os
import json
from collections import defaultdict

TRAIN_PATH = "../data/WikiQACorpus/train_rep.json"
DEV_PATH = "../data/WikiQACorpus/dev_rep.json"
TEST_PATH = "../data/WikiQACorpus/test_rep.json"
VOCAB = "../work/word2vec_vocab.txt"

def main():
    # read word2vec vocabs
    with open(VOCAB, 'r') as fi:
        vocabs = {x.strip().lower():1 for x in fi}

    # collect data vocabs
    files = [TRAIN_PATH, DEV_PATH, TEST_PATH]
    d = defaultdict(int)
    for f in files:
        with open(f, 'r') as fi:
            for line in fi:
                data = json.loads(line)
                for token in data['question']:
                    d[token.lower()] += 1
                for token in data['answer']:
                    d[token.lower()] += 1

    count = 0
    for token in d.keys():
        if token.lower() in vocabs:
            count += 1
        else:
            print(token)
    print("Total:{}\tFound:{}".format(len(d), count))

if __name__ == "__main__":
    main()
