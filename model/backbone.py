from torch import nn
from model.layers import DenseLayer


class BackBoneNet(nn.Module):
    def __init__(self, hidden_dim, num_layers, device):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(inSize=hidden_dim, outSize=hidden_dim,
                                                activation=nn.ReLU(), device=device)
                                     for _ in range(num_layers)])
        self.numLayers = num_layers

    def freeze(self, unfreeze=False):
        for layer in self.layers:
            layer.freeze(unfreeze)

    def forward(self, X):
        y = X
        for layer in self.layers:
            y = layer(y)
        return y
