# -*- coding: utf-8 -*-
"""
pklをロード
train/dev/testを標準入力から受け取って
QID\tLabel\tCNT\WGT_CNT
を返すスクリプト
"""
import sys
import json
import pickle

def main(fi):
    id2feature = pickle.load(open("../../work/features.pkl", "rb"))

    for line in fi:
        data = json.loads(line)
        wordcnt = id2feature[(data["question_id"], data["sentence_id"])]['wordcnt']
        wgt_wordcnt = id2feature[(data["question_id"], data["sentence_id"])]['wgt_wordcnt']
        label = data["label"]

        print("{}\t{}\t{}\t{}".format(
            data['question_id'], label, wordcnt, wgt_wordcnt
        ))

if __name__ == "__main__":
    main(sys.stdin)
