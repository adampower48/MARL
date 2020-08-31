import json

import numpy as np


def one_hot(idx, size, dtype=None):
    x = np.zeros(size, dtype=dtype)
    x[idx] = 1
    return x


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def write_json(js, filename):
    with open(filename, "w") as f:
        json.dump(js, f, indent=2)
