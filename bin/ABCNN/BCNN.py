# -*- coding: utf-8 -*-
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, Function, Variable, reporter
from chainer import Link, Chain
from .util import cos_sim, debug_print

class BCNN(Chain):

    def __init__(self, n_vocab, n_layer, embed_dim, input_channel, output_channel, train=True):
        self.train = train
        self.n_layer = n_layer
        # initialize all embeddings by uniform sampling.
        # but they are replaced by word2vec afterwards (except unknown token)
        if self.n_layer == 1:
            # 俺が今まで単層だと思ってたのは2層だったんだよ
            super(BCNN, self).__init__(
                embed=L.EmbedID(n_vocab, embed_dim, initialW=np.random.uniform(-0.01, 0.01)),  # 100: word-embedding vector size
                conv1=L.Convolution2D(
                    input_channel, output_channel, (4, embed_dim), pad=(3,0)),
                l1=L.Linear(in_size=2+4, out_size=1)  # 4 are from lexical features of WikiQA Task
            )
        elif self.n_layer == 2:
            super(BCNN, self).__init__(
                embed=L.EmbedID(n_vocab, embed_dim, initialW=np.random.uniform(-0.01, 0.01)),  # 100: word-embedding vector size
                conv1=L.Convolution2D(
                    input_channel, output_channel, (4, embed_dim), pad=(3,0)),
                conv2=L.Convolution2D(
                    input_channel, output_channel, (4, 50), pad=(3,0)),
                l1=L.Linear(in_size=2+4, out_size=1)  # 4 are from lexical features of WikiQA Task
            )


    def load_glove_embeddings(self, glove_path, vocab):
        assert self.embed != None
        print("loading GloVe vector...", end='', flush=True, file=sys.stderr)
        with open(glove_path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec
        print("done", flush=True, file=sys.stderr)

    def load_word2vec_embeddings(self, word2vec_path, vocab):
        assert self.embed != None
        print("loading word2vec vector...", end='', flush=True, file=sys.stderr)
        with open(word2vec_path, "r") as fi:
            for n, line in enumerate(fi):
                # 1st line contains stats
                if n == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line.strip().split(" ")[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec
        print("done", flush=True, file=sys.stderr)

    def __call__(self, x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len):
        x1_vecs = self.encode_sequence(x1s)
        x2_vecs = self.encode_sequence(x2s)
        # enc2 = self.encode_sequence(x2s)

        # similarity score for block 2 and 3 (block 1 is embedding layer)
        sim_scores = [F.squeeze(cos_sim(v1, v2), axis=2) for v1, v2 in zip(x1_vecs, x2_vecs)]

        feature_vec = F.concat(sim_scores + [wordcnt, wgt_wordcnt, x1s_len, x2s_len], axis=1)
        fc = F.squeeze(self.l1(feature_vec), axis=1)
        if self.train:
            return fc
        else:
            return fc, sim_scores


    def encode_sequence(self, xs):
        seq_length = xs.shape[1]
        # 1. wide_convolution
        embed_xs = self.embed(xs)
        batchsize, height, width = embed_xs.shape
        embed_xs = F.reshape(embed_xs, (batchsize, 1, height, width))
        embed_xs.unchain_backward()  # don't move word vector
        xs_conv1 = F.tanh(self.conv1(embed_xs))
        # (batchsize, depth, width, height)
        xs_conv1_swap = F.swapaxes(xs_conv1, 1, 3)  # (3, 50, 20, 1) --> (3, 1, 20, 50)
        # 2. average_pooling with window
        xs_avg = F.average_pooling_2d(xs_conv1_swap, ksize=(4, 1), stride=1, use_cudnn=False)
        assert xs_avg.shape[2] == seq_length  # average pooling語に系列長が元に戻ってないといけない

        embed_avg = F.average_pooling_2d(embed_xs, ksize=(embed_xs.shape[2], 1))
        xs_avg_1 = F.average_pooling_2d(xs_avg, ksize=(xs_avg.shape[2], 1))
        if self.n_layer == 1:
            # print(cos_sim(embed_avg, xs_avg_1).debug_print())
            return embed_avg, xs_avg_1
        elif self.n_layer == 2:
            xs_conv2 = F.tanh(self.conv2(xs_avg))
            xs_avg_2 = F.average_pooling_2d(xs_conv2, ksize=(xs_conv2.shape[2], 1))
            return embed_avg, xs_avg_1, xs_avg_2
