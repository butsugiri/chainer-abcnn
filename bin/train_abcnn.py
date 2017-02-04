# coding: utf-8
import os
import json
import sys
import argparse
from datetime import datetime
import numpy as np

import chainer
from chainer import reporter, training, cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions

from ABCNN.model import ABCNN
from ABCNN.updater import ABCNNUpdater
from ABCNN.classifier import Classifier
from ABCNN.wikiqa_evaluator import WikiQAEvaluator
from ABCNN.util import concat_examples, SelectiveWeightDecay, set_random_seed
from ABCNN.dev_iterator import DevIterator
from ABCNN.data_processor import DataProcessor
import BCNN.util


def main(args):
    start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    dest = "../result/" + start_time
    os.makedirs(dest)
    abs_dest = os.path.abspath(dest)
    with open(os.path.join(dest, "settings.json"), "w") as fo:
        fo.write(json.dumps(vars(args), sort_keys=True, indent=4))
        print(json.dumps(vars(args), sort_keys=True, indent=4), file=sys.stderr)

    # load data
    data_processor = DataProcessor(args.data, args.vocab, args.test, args.max_length)
    data_processor.prepare_dataset()
    data_processor.compute_max_length()
    train_data = data_processor.train_data
    dev_data = data_processor.dev_data
    test_data = data_processor.test_data


    # create model
    vocab = data_processor.vocab
    embed_dim = args.dim
    x1s_len = data_processor.max_x1s_len
    x2s_len = data_processor.max_x2s_len
    model_type = args.model_type
    if args.model_type == 'ABCNN1' or args.model_type == 'ABCNN3':
        input_channel = 2
    else:
        input_channel = 1
    cnn = ABCNN(n_vocab=len(vocab), embed_dim=embed_dim, input_channel=input_channel,
               output_channel=50, x1s_len=x1s_len, x2s_len=x2s_len, model_type=model_type, single_attention_mat=args.single_attention_mat)  # ABCNNはoutput = 50固定らしいが．
    model = Classifier(cnn, lossfun=sigmoid_cross_entropy,
                         accfun=binary_accuracy)
    if args.glove:
        cnn.load_glove_embeddings(args.glove_path, data_processor.vocab)
    if args.word2vec:
        cnn.load_word2vec_embeddings(args.word2vec_path, data_processor.vocab)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    cnn.set_pad_embedding_to_zero(data_processor.vocab)

    # setup optimizer
    optimizer = O.AdaGrad(args.lr)
    optimizer.setup(model)
    # do not use weight decay for embeddings
    decay_params = {name: 1 for name,
                    variable in model.namedparams() if "embed" not in name}
    optimizer.add_hook(SelectiveWeightDecay(
        rate=args.decay, decay_params=decay_params))

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    dev_train_iter = chainer.iterators.SerialIterator(
        train_data, args.batchsize, repeat=False)
    if args.use_test_data:
        dev_iter = DevIterator(test_data, data_processor.n_test)
    else:
        dev_iter = DevIterator(dev_data, data_processor.n_dev)

    x1s_len = np.array([cnn.x1s_len], dtype=np.int32)
    x2s_len = np.array([cnn.x2s_len], dtype=np.int32)
    updater = ABCNNUpdater(train_iter, optimizer, converter=BCNN.util.concat_examples, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=abs_dest)

    # setup evaluation
    eval_predictor = model.copy().predictor
    eval_predictor.train = False
    iters = {"train": dev_train_iter, "dev": dev_iter}
    trainer.extend(WikiQAEvaluator(
        iters, eval_predictor, converter=BCNN.util.concat_examples, device=args.gpu))

    # extentions...
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'validation/main/map', 'validation/main/mrr', 'validation/main/svm_map', 'validation/main/svm_mrr']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # take a shapshot when the model achieves highest accuracy in dev set
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}',
        trigger=chainer.training.triggers.MaxValueTrigger('validation/main/map')))
    # trainer.extend(extensions.ExponentialShift("lr", 0.5, optimizer=optimizer),
    #                trigger=chainer.training.triggers.MaxValueTrigger("validation/main/map"))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,
                        default=-1, help='GPU ID (Negative value indicates CPU)')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        default=5, help='Number of times to iterate through the dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=32, help='Minibatch size')
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
    parser.add_argument('--lr',  type=float,
                        default=0.08, help='Learning rate')
    parser.add_argument('--max-length', dest="max_length", type=int,
                        default=40, help='Max length of the sentence. (longer sentence gets truncated)')
    parser.add_argument('--single-attention-mat', dest="single_attention_mat", action='store_true',
                        help='Use same matrix for attention')
    parser.add_argument('--model-type', dest="model_type", type=str, default='ABCNN1',
                        help='Model type')
    parser.add_argument('--use-test-data', dest="use_test_data", action='store_true',
                        help='Use test data instead of dev data')
    parser.set_defaults(use_test_data=False)

    args = parser.parse_args()

    # can't use word2vec and glove at the same time
    assert not (args.glove and args.word2vec)
    assert args.model_type in ["ABCNN1", "ABCNN2", "ABCNN3"]
    main(args)
