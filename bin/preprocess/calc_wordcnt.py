# -*- coding: utf-8 -*-
"""
標準入力から1行1jsonを受け取って
IDFの重み付きWordCountと，Vanilla WordCountを付与したjsonを吐く．
出力されるjsonはQuestionIDとSentencedIDをkeyとするハッシュ
"""
import sys
import json
import pickle

def main(fi):
    word2idf = json.load(open("../../work/idf.json"))
    stopwords = {x:1 for x in json.load(open("../../work/stopwords.txt"))}

    id2wordcnt = {}
    for line in fi:
        data = json.loads(line)
        common_words = set(data["question"]) & set(data["answer"])
        common_nonstop_words = [word for word in common_words if word not in stopwords]
        wordcnt = len(common_nonstop_words)
        wgt_wordcnt = sum(word2idf[word] for word in common_nonstop_words if word in word2idf)
        id2wordcnt[(data['question_id'], data['sentence_id'])] = {"wordcnt": wordcnt, "wgt_wordcnt": wgt_wordcnt}
    with open("../../work/features.pkl", "wb") as fo:
        pickle.dump(id2wordcnt, fo)

if __name__ == "__main__":
    main(sys.stdin)
