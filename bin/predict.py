# coding: utf-8
import os
import json
import sys
import argparse
import numpy as np
from datetime import datetime
from itertools import groupby

import chainer
from chainer import reporter, training
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
import chainer.serializers as S
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions

from ABCNN import BCNN, DataProcessor, concat_examples, DevIterator, WikiQAEvaluator, SelectiveWeightDecay, compute_map_mrr


def main(config):
    # load data
    data_processor = DataProcessor(config["data"], config["vocab"], config["test"])
    data_processor.prepare_dataset()

    # select data
    if config['data_type'] == 'train':
        data = data_processor.train_data
        n_pair = data_processor.n_train
        json_path = "../data/WikiQACorpus/train.json"
    elif config['data_type'] == 'dev':
        data = data_processor.dev_data
        n_pair = data_processor.n_dev
        json_path = "../data/WikiQACorpus/dev.json"
    elif config['data_type'] == 'test':
        data = data_processor.test_data
        n_pair = data_processor.n_test
        json_path = "../data/WikiQACorpus/test.json"

    # iteratorからquestion_idを吐かせるのがダルいので…
    with open(json_path, 'r') as fi:
        q_id_pair = [(q_id, len(list(section))) for q_id, section in groupby(fi, key=lambda x: json.loads(x)["question_id"])]

    # create model
    vocab = data_processor.vocab
    embed_dim = config["dim"]
    cnn = BCNN(n_vocab=len(vocab), embed_dim=embed_dim, input_channel=1, n_layer=config["layer"],
               output_channel=50)  # ABCNNはoutput = 50固定らしいが．

    if config['glove']:
        cnn.load_glove_embeddings(config["glove_path"], data_processor.vocab)
    if config['word2vec']:
        cnn.load_word2vec_embeddings(config["word2vec_path"], data_processor.vocab)
    model = L.Classifier(cnn, lossfun=sigmoid_cross_entropy,
                         accfun=binary_accuracy)
    S.load_npz(config["model"], model)

    if config["gpu"] >= 0:
        model.to_gpu()

    predictor = model.predictor
    predictor.train = False
    predict_iter = DevIterator(data, n_pair)

    label_scores = []
    for n, batch in enumerate(predict_iter):
        # match question_id and prediction
        assert q_id_pair[n][1] == len(batch)
        x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len, y = concat_examples(batch)
        y_score, sim_scores = predictor(x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len)
        label_score = np.c_[y, y_score.data]
        label_scores.append(label_score)
    stats = compute_map_mrr(label_scores=label_scores)
    print("\nResults of direct evaluation from (A)BCNN")
    print("MAP: {map}\tMRR: {mrr}".format(
        map=stats.map,
        mrr=stats.mrr
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str,
                        required=True, help='Trained model file')
    parser.add_argument('--datatype',  type=str,
                        required=True, help='data_type')
    args = parser.parse_args()

    model_dir = os.path.dirname(args.model)
    # load config file, which is created in the training process
    config_path = os.path.join(model_dir, "settings.json")
    config = json.load(open(config_path, 'r'))
    config["model"] = args.model
    config["data_type"] = args.datatype

    main(config)
