import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

INPUT_DIM_SINGLE = 12
INPUT_DIM_STACK = 24
ACTION_DIM = 2
LATENT_DIM = 128
SHARED_ENC_DIM = 64


def getTotalActivation(act):
    return np.sum(np.abs(act))


class DenseLayer(nn.Module):
    def __init__(self, inSize, outSize, activation, device, batch_norm=False, is_shared=False):
        super(DenseLayer, self).__init__()
        self.is_shared = is_shared
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
        self.activation_out = getTotalActivation(y.detach().cpu().numpy())
        return y

    def freeze(self, unfreeze=False):
        if not unfreeze:  # Freeze params.
            for param in self.parameters():
                param.requires_grad = False
            self.isModuleFrozen = True
        else:  # Unfreeze params.
            for param in self.parameters():
                param.requires_grad = True
            self.isModuleFrozen = False


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, activation, device, is_shared=False):
        super(MLP, self).__init__()
        self.is_shared = is_shared
        input_layer = DenseLayer(inSize=input_dim, outSize=hidden_dim,
                                 activation=activation, device=device, is_shared=is_shared)
        layers = [DenseLayer(inSize=hidden_dim, outSize=hidden_dim,
                             activation=activation, device=device, is_shared=is_shared)
                  for _ in range(num_layers - 2)]
        output_layer = DenseLayer(inSize=hidden_dim, outSize=out_dim,
                                  activation=activation, device=device, is_shared=is_shared)
        self.layers = nn.Sequential(input_layer, *layers, output_layer)
        self.numLayers = num_layers
        self.isModuleFrozen = False

    def freeze(self, unfreeze=False):
        for layer in self.layers:
            layer.freeze(unfreeze)
        self.isModuleFrozen = False

    def forward(self, X):
        return self.layers(X)


class MultiTaskAttention(nn.Module):
    def __init__(self, latent_dim, is_shared=False):
        super(MultiTaskAttention, self).__init__()
        self.is_shared = is_shared
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.isModuleFrozen = False

    def freeze(self, unfreeze=False):
        if not unfreeze:  # Freeze params.
            for param in self.parameters():
                param.requires_grad = False
            self.isModuleFrozen = True
        else:  # Unfreeze params.
            for param in self.parameters():
                param.requires_grad = True
            self.isModuleFrozen = False

    def forward(self, x):
        # x shape: [batch_size, seq_len=1, LATENT_DIM]
        x = x.unsqueeze(1)  # Adding sequence length dimension
        attn_output, _ = self.attn(x, x, x)
        return attn_output.squeeze(1)  # Output shape: [batch_size, LATENT_DIM]
