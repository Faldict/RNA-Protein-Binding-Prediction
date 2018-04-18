from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def onehot_encoding(seq, length=300):
    """ Encode RNA sequence to vector
    """
    assert len(seq) == length

    rna = {
        'A': (1, 0, 0, 0),
        'G': (0, 1, 0, 0),
        'C': (0, 0, 1, 0),
        'T': (0, 0, 0, 1)
    }
    vec = [rna[c] for c in seq]
    return np.array(vec)

def load_dataset(rbp, filepath="RNA_trainset/", filename="train"):
    """ load dataset
    """
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    with open(os.path.join(filepath, rbp, filename)) as f:
        lines = f.readlines()
        num = [0, 0]
        for line in lines:
            seq, label = line.split()
            label = int(label)
            if num[label] < 400:
                test_data.append(onehot_encoding(seq))
                test_labels.append(label)
                num[label] += 1
            else:
                train_data.append(onehot_encoding(seq))
                train_labels.append(label)
    return np.array(train_data, dtype=np.float32), np.array(train_labels, dtype=np.int32), np.array(test_data, dtype=np.float32), np.array(test_labels, dtype=np.int32)
