# coding: utf-8
import numpy as np
import sys
import argparse
import copy
from collections import defaultdict
from sklearn.metrics import average_precision_score

import chainer
from chainer import reporter, training
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions

from ABCNN import BCNN, DataProcessor, concat_examples, DevIterator, WikiQAEvaluator


def main(args):
    data_processor = DataProcessor(args.data, args.test)
    data_processor.prepare_dataset()
    train_data = data_processor.train_data
    dev_data = data_processor.dev_data

    vocab = data_processor.vocab
    embed_dim = args.dim
    cnn = BCNN(n_vocab=len(vocab), embed_dim=embed_dim, input_channel=1,
               output_channel=50)  # ABCNNはoutput = 50固定らしいが．
    if args.glove:
        cnn.load_glove_embeddings(args.glove_path, data_processor.vocab)
    model = L.Classifier(cnn, lossfun=sigmoid_cross_entropy,
                         accfun=binary_accuracy)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    dev_iter = DevIterator(dev_data, data_processor.n_dev)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=concat_examples, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # Evaluation
    eval_predictor = model.copy().predictor
    trainer.extend(WikiQAEvaluator(
        dev_iter, eval_predictor, converter=concat_examples, device=args.gpu))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'validation/main/average_precision', 'validation/main/reciprocal_rank']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    # trainer.extend(extensions.snapshot_object(
    #     model, 'model_epoch_{.updater.epoch}',
    #     trigger=chainer.training.triggers.MaxValueTrigger('validation/main/average_precision')))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='negative value indicates CPU')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=3, help='learning minibatch size')
    parser.add_argument('--glove_path', dest='glove_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Pretrained glove vector')
    parser.add_argument('--data',  type=str,
                        default='../data/WikiQACorpus', help='path to data file')
    parser.add_argument('--dim',  type=int,
                        default=10, help='embedi dimension')
    # parser.add_argument('--glove', action='store_true', help='use GloVe vector?')
    parser.set_defaults(glove=False)
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.set_defaults(test=False)

    args = parser.parse_args()
    main(args)
