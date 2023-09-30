import numpy as np


def prepare_vocab(vocab):
    classes = []
    mapped_classes = []
    if isinstance(vocab, dict):
        vocab = [vocab]
    for x in vocab:
        if isinstance(x, dict):
            for c, label in x.items():
                classes.append(c)
                mapped_classes.append(label)
        else:
            classes.append(x)
            mapped_classes.append(x)
    return np.array(classes), np.array(mapped_classes)