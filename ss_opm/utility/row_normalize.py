import numpy as np


def row_normalize(v):
    mu = np.mean(v, axis=1)
    sigma = np.std(v, axis=1)
    return (v - mu[:, None]) / sigma[:, None]
