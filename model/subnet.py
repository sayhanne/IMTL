import torch.nn as nn
from model.layers import DenseLayer


class SubNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, device):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(inSize=hidden_dim, outSize=hidden_dim,
                                                activation=nn.ReLU(), device=device)
                                     for _ in range(num_layers - 1)])
        self.layers.append(DenseLayer(inSize=hidden_dim, outSize=out_dim, activation=None, device=device))
        self.numLayers = num_layers

    def freeze(self, unfreeze=False):
        for layer in self.layers:
            layer.freeze(unfreeze)

    def forward(self, X):
        y = X
        for layer in self.layers:
            y = layer(y)
        return y
