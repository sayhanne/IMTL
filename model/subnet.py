import torch.nn as nn
from model.blocks import DenseLayer


class SubNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_layers, activation, device):
        super(SubNet, self).__init__()
        layers = [DenseLayer(inSize=hidden_dim, outSize=hidden_dim,
                             activation=activation, device=device)
                  for _ in range(num_layers - 1)]
        layers.append(DenseLayer(inSize=hidden_dim, outSize=out_dim, activation=None, device=device))
        self.layers = nn.Sequential(*layers)
        self.numLayers = num_layers

    def freeze(self, unfreeze=False):
        for layer in self.layers:
            layer.freeze(unfreeze)

    def forward(self, X):
        return self.layers(X)
