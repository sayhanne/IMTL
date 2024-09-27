import numpy as np
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, inSize, outSize, activation, device, batch_norm=False):
        super(DenseLayer, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.device = device
        self.module = nn.Linear(inSize, outSize, device=self.device)
        if batch_norm:
            self.bn = nn.BatchNorm1d(outSize)
        self.batch_norm = batch_norm
        self.isModuleFrozen = False
        self.activation_out = None

        if activation is None:
            self.activation = (lambda x: x)
        else:
            self.activation = activation

    def forward(self, X):
        z = self.module(X)
        if self.batch_norm:
            if self.isModuleFrozen:
                self.bn.eval()
            else:
                self.bn.train()
            y = self.activation(self.bn(z))
        else:
            y = self.activation(z)
        self.activation_out = self.getTotalActivation(y.detach().cpu().numpy())
        return y

    def getTotalActivation(self, act):
        return np.sum(np.abs(act))

    def freeze(self, unfreeze=False):
        if not unfreeze:  # Freeze params.
            for param in self.parameters():
                param.requires_grad = False
            self.isModuleFrozen = True
        else:  # Unfreeze params.
            for param in self.parameters():
                param.requires_grad = True
            self.isModuleFrozen = False
