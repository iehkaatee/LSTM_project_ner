"""
useful functions
"""

import torch
import numpy as np

def prepare_sequence(seq, to_ix):
    """
        translate words to idx and return a torch tensor.
        params:    seq      sequence of words to translate
                   to_ix    translation dictionary
    """

    idxs = [to_ix[w] for w in seq]

    return torch.tensor(idxs, dtype=torch.long)

def load_data(path, data_list, as_list=True):
    """
        load data from @path to @data_list.
    """
    if as_list:
        with open(path) as f:
            for l in f.read().splitlines():
                data_list.append(l.split(' '))
    else:
        with open(path) as f:
            for l in f.read().splitlines():
                data_list.append(l)

def append_to_vocab(data, word_to_ix, label_to_ix, pos_to_ix):
    """
        create and increase the vocabulary with data being used.
    """
    for sentence, pos, labels in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for label in labels:
            if label not in label_to_ix:
                label_to_ix[label] = len(label_to_ix)
        for tag in pos:
            if tag not in pos_to_ix:
                pos_to_ix[tag] = len(pos_to_ix)
