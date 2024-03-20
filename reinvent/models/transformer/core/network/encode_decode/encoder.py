import torch.nn as nn

from reinvent.models.transformer.core.network.encode_decode.clones import clones
from reinvent.models.transformer.core.network.encode_decode.layer_norm import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
