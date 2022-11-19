import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):

    def __init__(self, n_channels=128, length=32, kernel_size=3, dropout_p=0.1, norm="batch_norm", skip=False):
        super(ConvBlock, self).__init__()
        bias = norm == "none"
        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding="same",
        )
        self.skip = skip
        if norm == "batch_norm":
            self.norm = nn.BatchNorm1d(n_channels)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        elif norm == "layer_nome":
            self.norm = nn.LayerNorm((n_channels, length))
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        else:
            self.norm = None
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None
        self.act = nn.ReLU()

    def forward(self, x):
        h = x
        h = self.act(h)
        if self.norm is not None:
            h = self.norm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.conv(h)
        return h

class ConvResidualBlock(nn.Module):

    def __init__(self, n_channels=128, length=32, kernel_size=3, dropout_p=0.1, norm="batch_norm", skip=False):
        super(ConvResidualBlock, self).__init__()
        self.skip = skip
        self.conv_block1 = ConvBlock(
            length=length,
            n_channels=n_channels,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            norm=norm,
            skip=self.skip
        )
        self.conv_block2 = ConvBlock(
            length=length,
            n_channels=n_channels,
            kernel_size=1,
            dropout_p=dropout_p,
            norm=norm,
            skip=False
        )

    def forward(self, x):
        h = x
        h_prev = x
        h = self.conv_block1(h)
        h = self.conv_block2(h)
        if self.skip:
            h = h + h_prev
        return h

class CNNModule(nn.Module):

    def __init__(self, input_dim, output_dim, n_block, length=32, n_channels=8, skip=False, dropout_p=0.1, norm="batch_norm"):
        super(CNNModule, self).__init__()
        self.length = length
        self.n_channels = n_channels
        if input_dim is not None:
            self.in_fc =  nn.LazyLinear(out_features=self.length*self.n_channels)
        else:
            self.in_fc = None
        layers = []
        for _ in range(n_block):
            layers.append(ConvResidualBlock(n_channels=n_channels, kernel_size=3, skip=skip, dropout_p=dropout_p, norm=norm, length=length))
        self.layers = nn.ModuleList(layers)
        if output_dim is not None:
            self.out_fc = nn.LazyLinear(out_features=output_dim)
        else:
            self.out_fc = None

    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        h = h.reshape(-1, self.n_channels, self.length)
        for l in self.layers:
            h = l(h)
        h = torch.flatten(h, start_dim=1)
        if self.out_fc is not None:
            h = self.out_fc(h)
        y = h
        return y

