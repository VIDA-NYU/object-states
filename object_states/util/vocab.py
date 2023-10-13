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
            xs = [x.strip() for x in x.split(':', 1)]
            classes.append(xs[0])
            mapped_classes.append(xs[-1])
    return np.array(classes), np.array(mapped_classes)