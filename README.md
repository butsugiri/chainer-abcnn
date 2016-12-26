# Introduction
Re-implementation of `Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).`

# Requirements
* Python 3.*
* Chainer ver. 1.19.0 (or higher)
* Dataset to train the model (I used [Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/) )
* Pretrained GloVe vector (Get them from [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/projects/glove/))


# Data format for input data
- [0 or 1] [Sequence of words]
    - 1 and 0 are positive and negative, respectively.
- There is a python script that splits data into train/dev/test set.
    - Run `python split_train_dev_split < YOURDATA`


## Examples
```
1 That was so beautiful that it can't be put into words . (POSITIVE SETENCE)
0 I do not want to go to school because I do like to study math . (NEGATIVE SENTENCE)
```

# Usage
```
  $ python train_cnn.py
```

# Optional arguments
```
-h, --help            show this help message and exit
--gpu   GPU           negative value indicates CPU
--epoch EPOCH         number of epochs to learn
--batchsize BATCHSIZE
                      learning minibatch size
--nunits NUNITS       number of units
--glove GLOVE_PATH    Pretrained glove vector
--test                use tiny dataset
```
