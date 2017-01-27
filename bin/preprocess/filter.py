# -*- coding: utf-8 -*-
"""
for WikiQA
this script removes questions that does not have
right answer in it.
"""
import sys
import argparse
from itertools import groupby

def main(fi, data_type):
    filter_doc_path = "../../data/WikiQACorpus/WikiQA-{}-filtered.ref".format(data_type)
    targets = {x.split()[0]:1 for x in open(filter_doc_path)}

    for line in fi:
        line = line.strip()
        question_id = line.split()[0]
        if question_id in targets:
            print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hogehoge")
    parser.add_argument('--data', default='train', type=str, help='data type')
    args = parser.parse_args()
    main(sys.stdin, args.data)
