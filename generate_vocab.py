# -*- coding: utf-8 -*-
import argparse
import sys
import json
from collections import defaultdict

def main(fi, threshold):
    token_freqs = defaultdict(int)
    for line in fi:
        tokens = line.strip().split(" ")[1::]
        for token in tokens:
            token_freqs[token] += 1

    token2id = defaultdict(lambda: len(token2id))
    for token, freq in token_freqs.items():
        if freq <= threshold:
            continue
        elif token.strip() == "":
            continue
        else:
            token2id[token]

    for vocab, _id in token2id.items():
        print("{}\t{}".format(vocab, _id))

    sys.stderr.write("Threshold Value: {}\nOriginal Vocab Size:{}\tVocab Size (After Cut-off):{}\n".format(
        threshold,
        len(token_freqs),
        len(token2id),
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Vocab creator")
    parser.add_argument('-t', '--threshold', dest='threshold', default=2, type=int,help='しきい値')
    args = parser.parse_args()
    main(sys.stdin, args.threshold)
