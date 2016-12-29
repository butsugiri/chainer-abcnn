# -*- coding: utf-8 -*-
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, Function, Variable
from chainer import Link, Chain


class BCNN(Chain):

    def __init__(self, n_vocab, input_channel, output_channel, n_label, train=True):
        super(BCNN, self).__init__(
            embed=L.EmbedID(n_vocab, 100),  # 100: word-embedding vector size
            conv1=L.Convolution2D(
                input_channel, output_channel, (4, 100), pad=(3,0)),
            conv2=L.Convolution2D(
                input_channel, output_channel, (4, 50), pad=(3,0)),
            norm1=L.BatchNormalization(1), # embedding用
            norm2=L.BatchNormalization(50), # conv1用
            norm3=L.BatchNormalization(50), # conv2用
            l1=L.Linear(4 * 50, n_label)
        )
        self.train = train

    def load_glove_embeddings(self, glove_path, vocab):
        assert self.embed != None
        sys.stderr.write("loading GloVe vector...")
        with open(glove_path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=np.float32)
                    self.embed.W.data[vocab[word]] = vec
        sys.stderr.write("done\n")

    def __call__(self, x1s, x2s):
        enc1 = self.encode_sequence(x1s)
        enc2 = self.encode_sequence(x2s)
        concat = F.concat([enc1, enc2], axis=1)
        return F.tanh(self.l1(concat))

    def encode_sequence(self, xs):
        seq_length = xs.shape[1]
        # 1. wide_convolution
        embed_xs = self.embed(xs)
        batchsize, height, width = embed_xs.shape
        embed_xs = F.reshape(embed_xs, (batchsize, 1, height, width))
        embed_xs = self.norm1(embed_xs)
        xs_conv1 = F.tanh(self.norm2(self.conv1(embed_xs)))
        # (batchsize, depth, width, height)
        xs_all_avg_2 = F.average_pooling_2d(xs_conv1, ksize=(xs_conv1.shape[2], 1))
        xs_conv1_swap = F.swapaxes(xs_conv1, 1, 3) # (3, 50, 20, 1) --> (3, 1, 20, 50)

        # 2. average_pooling with window
        xs_avg = F.average_pooling_2d(xs_conv1_swap, ksize=(4, 1), stride=1, use_cudnn=False)
        assert xs_avg.shape[2] == seq_length # average pooling語に系列長が元に戻ってないといけない

        # 3. wide_convolution
        xs_conv2 = F.tanh(self.norm3(self.conv2(xs_avg)))

        # 4. all_average_pooling
        xs_all_avg_1 = F.average_pooling_2d(xs_conv2, ksize=(xs_conv2.shape[2], 1))
        xs_all_avg_2 = F.average_pooling_2d(xs_conv1, ksize=(xs_conv1.shape[2], 1))
        return F.concat([xs_all_avg_1, xs_all_avg_2], axis=1)
