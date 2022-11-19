import torch
from torch import nn


class PredictionModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation="gelu", norm="layer_norm"):
        super().__init__()
        if norm == "layer_norm":
            norm_class = nn.LayerNorm
        else:
            raise RuntimeError
        if activation == "gelu":
            activation_class = nn.GELU
        else:
            raise RuntimeError
        layers = [
            nn.Linear(input_dim, hidden_dim),
            norm_class(hidden_dim),
            activation_class(),
            nn.Linear(hidden_dim, output_dim)
        ]
        self.module_list = nn.ModuleList(layers)

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x


class SimSiamModule(nn.Module):
    def __init__(self, prediction_module):
        super().__init__()
        self.prediction_module = prediction_module
        self.criterion = SymmetricNegativeCosineSimilarityLoss()

    def _forward(self, z):
        p = self.prediction_module(z)
        return p, z.detach()

    def forward(self, z0, z1):
        return self._forward(z0), self._forward(z1)

    def loss(self, z0, z1):
        out0, out1 = self(z0, z1)
        return self.criterion(out0, out1)


class SymmetricNegativeCosineSimilarityLoss(torch.nn.Module):

    def forward(self, x0, x1):
        p0, z0 = x0
        p1, z1 = x1
        #print("p0", p0)
        #print("z1", z1)
        negative_similarity0 = -torch.nn.functional.cosine_similarity(p0, z1, dim=1).mean()
        negative_similarity1 = -torch.nn.functional.cosine_similarity(p1, z0, dim=1).mean()
        negative_similarity = (negative_similarity0 + negative_similarity1)/2.0
        return negative_similarity