from torch import nn


class LinearBlock(nn.Module):
    def __init__(self, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="none"):
        super(LinearBlock, self).__init__()
        self.skip = skip
        self.fc = nn.Linear(h_dim, h_dim, bias=False)
        if norm == "batch_norm":
            self.norm = nn.BatchNorm1d(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        elif norm == "layer_nome":
            self.norm = nn.LayerNorm(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        else:
            self.norm = None
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            # print("activation", activation)
            self.act = nn.GELU()
        else:
            raise RuntimeError()

    def forward(self, x):
        h = x
        h_prev = x
        h = self.act(h)
        if self.norm is not None:
            h = self.norm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.fc(h)
        if self.skip:
            h = h + h_prev
        return h


class MLPBModule(nn.Module):
    def __init__(self, input_dim, output_dim, n_block, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="bn"):
        super(MLPBModule, self).__init__()
        self.in_fc = None
        if input_dim is not None:
            self.in_fc = nn.Linear(input_dim, h_dim)
        layers = []
        for _ in range(n_block):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p, activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        self.out_fc = None
        if output_dim is not None:
            self.out_fc = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        for layer in self.layers:
            h = layer(h)
        if self.out_fc is not None:
            y = self.out_fc(h)
        else:
            y = h
        return y, h


class HierarchicalMLPBModule(nn.Module):
    def __init__(self, input_dim, output_dim, n_block, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="bn"):
        super(HierarchicalMLPBModule, self).__init__()
        self.in_fc = None
        if input_dim is not None:
            self.in_fc = nn.Linear(input_dim, h_dim)
        layers = []
        for _ in range(n_block):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p, activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        self.out_fc = None
        if output_dim is not None:
            self.out_fc = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        hs = [h]
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        if self.out_fc is not None:
            y = self.out_fc(h)
        else:
            y = h
        return y, hs
