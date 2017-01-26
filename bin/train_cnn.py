# coding: utf-8
import os
import json
import sys
import argparse
from datetime import datetime

import chainer
from chainer import reporter, training
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions

from ABCNN import BCNN, DataProcessor, concat_examples, DevIterator, WikiQAEvaluator, SelectiveWeightDecay


def main(args):
    start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    dest = "../result/" + start_time
    os.makedirs(dest)
    abs_dest = os.path.abspath(dest)
    with open(os.path.join(dest, "settings.json"), "w") as fo:
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))

    # load data
    data_processor = DataProcessor(args.data, args.vocab, args.test)
    data_processor.prepare_dataset()
    train_data = data_processor.train_data
    dev_data = data_processor.dev_data

    # create model
    vocab = data_processor.vocab
    embed_dim = args.dim
    cnn = BCNN(n_vocab=len(vocab), embed_dim=embed_dim, input_channel=1,
               output_channel=50)  # ABCNNはoutput = 50固定らしいが．
    if args.glove:
        cnn.load_glove_embeddings(args.glove_path, data_processor.vocab)
    if args.word2vec:
        cnn.load_word2vec_embeddings(args.word2vec_path, data_processor.vocab)
    model = L.Classifier(cnn, lossfun=sigmoid_cross_entropy,
                         accfun=binary_accuracy)
    if args.gpu >= 0:
        model.to_gpu()

    # setup optimizer
    optimizer = O.AdaGrad(lr=0.08)
    optimizer.setup(model)
    # do not use weight decay for embeddings
    decay_params = {name: 1 for name,
                    variable in model.namedparams() if "embed" not in name}
    optimizer.add_hook(SelectiveWeightDecay(
        rate=args.decay, decay_params=decay_params))

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    dev_train_iter = chainer.iterators.SerialIterator(
        train_data, args.batchsize, repeat=False)
    dev_iter = DevIterator(dev_data, data_processor.n_dev)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=concat_examples, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=abs_dest)

    # setup evaluation
    eval_predictor = model.copy().predictor
    eval_predictor.train = False
    iters = {"train": dev_train_iter, "dev": dev_iter}
    trainer.extend(WikiQAEvaluator(
        iters, eval_predictor, converter=concat_examples, device=args.gpu))

    # extentions...
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'validation/main/map', 'validation/main/mrr', 'validation/main/svm_map', 'validation/main/svm_mrr']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}',
        trigger=chainer.training.triggers.MaxValueTrigger('validation/main/map')))
    trainer.extend(extensions.ExponentialShift("lr", 0.5, optimizer=optimizer),
                   trigger=chainer.training.triggers.MaxValueTrigger("validation/main/map"))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='GPU ID (Negative value indicates CPU)')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=5, help='Number of times to iterate through the dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=3, help='Minibatch size')
    parser.add_argument('--data',  type=str,
                        default='../data/WikiQACorpus', help='Path to input (train/dev/test) data file')
    parser.add_argument('--dim',  type=int,
                        default=10, help='embed dimension')
    parser.add_argument('--glove', action='store_true',
                        help='Use GloVe vector?')
    parser.set_defaults(glove=False)
    parser.add_argument('--glove-path', dest='glove_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Path to pretrained glove vector')

    parser.add_argument('--word2vec', action='store_true',
                        help='Use word2vec vector?')
    parser.set_defaults(word2vec=False)
    parser.add_argument('--word2vec-path', dest='word2vec_path', type=str,
                        default="../../disco_parse/data/glove_model/glove.6B.100d.txt", help='Path to pretrained word2vec vector')

    parser.add_argument('--test', action='store_true',
                        help='Use tiny dataset for quick test')
    parser.set_defaults(test=False)
    parser.add_argument('--decay',  type=float,
                        default=0.0002, help='Weight decay rate')
    parser.add_argument('--vocab',  type=str,
                        default="../work/word2vec_vocab.txt", help='Vocabulary file')

    args = parser.parse_args()

    # can't use word2vec and glove at the same time
    assert not (args.glove and args.word2vec)
    main(args)
