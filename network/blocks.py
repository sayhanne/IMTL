import math
import os

import numpy as np
import torch


def getTotalActivation(act):
    return np.mean(np.abs(act))


class Avg(torch.nn.Module):
    def __init__(self, dims):
        super(Avg, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class DenseLayer(torch.nn.Module):
    """ linear layer with optional batch normalization. """

    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(DenseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.activation_out = None
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            y = self.batch_norm(y)
        self.activation_out = getTotalActivation(y.detach().cpu().numpy())
        return y

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)


class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """

    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        if activation is None:
            activation = (lambda x: x)
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(DenseLayer(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(DenseLayer(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name + ".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name + ".ckpt"))
        self.train().to(dv)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, std=None, bias=True,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.activation_out = None
        self.block = [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, bias=bias)]
        if batch_norm:
            self.block.append(torch.nn.BatchNorm2d(out_channels))
        self.block.append(torch.nn.ReLU())

        if std is not None:
            self.block[0].weight.data.normal_(0., std)
            self.block[0].bias.data.normal_(0., std)
        self.block = torch.nn.Sequential(*self.block)

    def forward(self, x):
        y = self.block(x)
        self.activation_out = getTotalActivation(y.detach().cpu().numpy())
        return y


class MultiHeadAttnLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads=1, num_layers=1):
        super(MultiHeadAttnLayer, self).__init__()
        self.attention_layers = torch.nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
                                                     for _ in range(num_layers)])

    def forward(self, query, key, value):
        # Sequentially pass query, key, and value through each attention layer
        for layer in self.attention_layers:
            attn_output, attn_weights = layer(query, key, value)
            # Update query, key, and value with the output for the next layer
            query = key = value = attn_output
        return attn_output  # Return the output of the last layer


def build_state_encoder(config, shared=False, task_idx=-1):
    if shared:
        out_dim = config["hidden_dim"]
    else:
        out_dim = config["rep_state"]
    if "cnn" in config:
        if config["cnn"]:
            L = len(config["filters"]) - 1
            stride = 2
            encoder = []
            for i in range(L):
                encoder.append(ConvBlock(in_channels=config["filters"][i],
                                         out_channels=config["filters"][i + 1],
                                         kernel_size=3, stride=1, padding=1, batch_norm=config["batch_norm"]))
                encoder.append(ConvBlock(in_channels=config["filters"][i + 1],
                                         out_channels=config["filters"][i + 1],
                                         kernel_size=3, stride=stride, padding=1, batch_norm=config["batch_norm"]))
            encoder.append(Avg([2, 3]))
            encoder.append(MLP([config["filters"][-1], out_dim]))
        else:
            encoder = [
                Flatten([1, 2, 3]),
                MLP(layer_info=[config["in_size"] ** 2] + [config["hidden_dim"]] * config["enc_depth_state"] + [
                    out_dim],
                    batch_norm=config["batch_norm"])]
    else:   # not cnn
        if shared:
            # there will be a projection layer before this mlp encoder
            encoder = [MLP(
                layer_info=[config["hidden_dim"]] * (config["enc_depth_state"] - 1) + [out_dim],
                batch_norm=config["batch_norm"])]
        else:
            encoder = [MLP(
                layer_info=[config["in_size"][task_idx]] + [config["hidden_dim"]] * (config["enc_depth_state"] - 1) + [
                    out_dim],
                batch_norm=config["batch_norm"])]

    encoder = torch.nn.Sequential(*encoder)
    return encoder
