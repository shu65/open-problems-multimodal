import numpy as np
import torch


def targets_values_normalize(v):
    mu = np.mean(v, axis=1)
    sigma = np.std(v, axis=1)
    return (v - mu[:, None]) / sigma[:, None]


def targets_values_normalize_torch(v):
    mu = torch.mean(v, dim=1)
    sigma = torch.std(v, dim=1)
    return (v - mu[:, None]) / sigma[:, None]
