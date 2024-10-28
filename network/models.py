from torch import nn


class MTL(nn.Module):
    def __init__(self, config):
        super(MTL, self).__init__()
