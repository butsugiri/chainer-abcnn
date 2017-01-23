# -*- coding: utf-8 -*-
"""
Calculate IDF value from question corpus
"""
import sys
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def main(fi):
    questions = [" ".join(json.loads(line)['question']) for line in fi]
    model = TfidfVectorizer(stop_words="english")
    model.fit(questions)

    word2idf = {word:model.idf_[idx] for word, idx in model.vocabulary_.items()}
    sys.stderr.write("Saving IDF...")
    with open("../../work/idf.json", 'w') as fo:
        fo.write(json.dumps(word2idf))
    sys.stderr.write("Done.\n")

    sys.stderr.write("Saving StopWords...")
    with open("../../work/stopwords.txt", 'w') as fo:
        fo.write(json.dumps(list(model.get_stop_words())))
    sys.stderr.write("Done.\n")


if __name__ == "__main__":
    main(sys.stdin)
