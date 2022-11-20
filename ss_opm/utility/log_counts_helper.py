import numpy as np


def inverse_normalize(values, library_size):
    counts = np.expm1(values) / 1e6 * library_size
    return counts


def normalize(counts):
    total_size = np.sum(counts)
    return np.log1p(counts / total_size * 1e6)
