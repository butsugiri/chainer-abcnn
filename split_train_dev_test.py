# -*- coding: utf-8 -*-
import sys
import random

def main(fi):
    whole_data = [x.strip() for x in fi]
    random.seed(0)
    random.shuffle(whole_data)

    total_length = len(whole_data)
    n_train = int(total_length * 0.8)
    n_dev = n_train + int(total_length * 0.1)

    with open("./data/train.txt", "w") as train_data, open("./data/dev.txt", "w") as dev_data, open("./data/test.txt", "w") as test_data:
        for n, line in enumerate(whole_data):
            if 0 <= n <= n_train:
                train_data.write("{}\n".format(line))
            elif n_train <= n <= n_dev:
                dev_data.write("{}\n".format(line))
            else:
                test_data.write("{}\n".format(line))

if __name__ == "__main__":
    main(sys.stdin)
