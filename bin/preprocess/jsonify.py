# -*- coding: utf-8 -*-
"""
WikiQAの中身をtokenizeしたものに置換するスクリプト
1行1jsonで吐きます
"""
import sys
import argparse
import json
from itertools import groupby

def main(fi):
    for line in fi:
        data = line.strip().split("\t")
        question_id = data[1]
        document_id = data[3]
        sentence_id = data[5]
        title = data[4]
        question = data[-3]
        answer = data[-2]
        label = data[-1]

        out = {
            'question_id': question_id,
            'document_id': document_id,
            'sentence_id': sentence_id,
            'title': title,
            'question': question.lower().split(" "),
            'answer': answer.lower().split(" "),
            'label': label
        }

        print(json.dumps(out))



if __name__ == "__main__":
    main(sys.stdin)
