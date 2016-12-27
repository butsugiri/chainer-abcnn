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
            embed=L.EmbedID(n_vocab, 10),  # 100: embedding vector size
            conv1=L.Convolution2D(
                input_channel, output_channel, (4, 10), pad=3),
            conv2=L.Convolution2D(
                input_channel, output_channel, (4, 10)),
            l1=L.Linear(3 * output_channel, n_label)
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
        exit()
        # enc2 = self.encode_sequence(x2s)
        # exit()

        batchsize, height, width = xs.shape
        xs = F.reshape(xs, (batchsize, 1, height, width))
        conv = self.conv4(xs)
        h1 = F.max_pooling_2d(F.relu(conv3_xs), conv3_xs.shape[2])
        h2 = F.max_pooling_2d(F.relu(conv4_xs), conv4_xs.shape[2])
        h3 = F.max_pooling_2d(F.relu(conv5_xs), conv5_xs.shape[2])
        concat_layer = F.concat([h1, h2, h3], axis=1)
        y = self.l1(F.dropout(concat_layer, train=self.train))
        return y

    def encode_sequence(self, xs):
        xs = self.embed(xs)
        batchsize, height, width = xs.shape
        xs = F.reshape(xs, (batchsize, 1, height, width))
        conv = self.conv1(xs)
        print(conv.data)
        print(xs.shape)
        print(conv.shape)
        pass
