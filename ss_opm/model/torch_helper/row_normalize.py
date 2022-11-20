import torch


def row_normalize(v):
    mu = torch.mean(v, dim=1)
    sigma = torch.std(v, dim=1)
    return (v - mu[:, None]) / sigma[:, None]
