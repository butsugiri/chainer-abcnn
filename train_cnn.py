# coding: utf-8
import numpy as np
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.training import extensions
import argparse

from chainer import training
from ABCNN import BCNN, DataProcessor, concat_examples


def main(args):
    data_path = "data/"
    data_processor = DataProcessor(data_path, args.test)
    data_processor.prepare_dataset()
    train_data = data_processor.train_data
    dev_data = data_processor.dev_data

    vocab = data_processor.vocab
    target = data_processor.connective
    cnn = BCNN(n_vocab=len(vocab), input_channel=1,
                  output_channel=50, n_label=len(target)) # ABCNNはoutput = 50固定らしいが．
    if args.glove:
        cnn.load_glove_embeddings(args.glove_path, data_processor.vocab)
    model = L.Classifier(cnn)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(dev_data, args.batchsize,
                                                repeat=False, shuffle=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=concat_examples, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # Evaluation
    eval_model = model.copy()
    eval_model.predictor.train = False
    trainer.extend(extensions.Evaluator(
        dev_iter, eval_model, device=args.gpu, converter=concat_examples))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}',
        trigger=chainer.training.triggers.MaxValueTrigger('validation/main/accuracy')))
    trainer.run()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='negative value indicates CPU')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=3, help='learning minibatch size')
    parser.add_argument('--glove_path', dest='glove_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Pretrained glove vector')
    # parser.add_argument('--glove', action='store_true', help='use GloVe vector?')
    parser.set_defaults(glove=False)
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.set_defaults(test=False)

    args=parser.parse_args()
    main(args)
