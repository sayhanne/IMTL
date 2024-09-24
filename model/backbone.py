from torch import nn
from model.layers import DenseLayer


class BackBoneNet(nn.Module):
    def __init__(self, hidden_dim, num_layers, activation, device):
        super(BackBoneNet, self).__init__()
        layers = [DenseLayer(inSize=hidden_dim, outSize=hidden_dim,
                             activation=activation, device=device)
                  for _ in range(num_layers)]
        self.layers = nn.Sequential(*layers)
        self.numLayers = num_layers

    def freeze(self, unfreeze=False):
        for layer in self.layers:
            layer.freeze(unfreeze)

    def forward(self, X):
        return self.layers(X)
