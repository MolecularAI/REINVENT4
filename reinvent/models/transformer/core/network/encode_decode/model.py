import copy

import torch.nn as nn

from reinvent.models.transformer.core.network.encode_decode.decoder import Decoder
from reinvent.models.transformer.core.network.encode_decode.decoder_layer import DecoderLayer
from reinvent.models.transformer.core.network.encode_decode.encoder import Encoder
from reinvent.models.transformer.core.network.encode_decode.encoder_layer import EncoderLayer
from reinvent.models.transformer.core.network.module.embeddings import Embeddings
from reinvent.models.transformer.core.network.module.generator import Generator
from reinvent.models.transformer.core.network.module.multi_headed_attention import (
    MultiHeadedAttention,
)
from reinvent.models.transformer.core.network.module.positional_encoding import (
    PositionalEncoding,
)
from reinvent.models.transformer.core.network.module.positionwise_feedforward import (
    PositionwiseFeedForward,
)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers=6,
        num_heads=8,
        model_dimension=256,
        feedforward_dimension=2048,
        dropout=0.1,
    ):
        super(EncoderDecoder, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dimension = model_dimension
        self.feedforward_dimension = feedforward_dimension
        self.dropout = dropout

        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, model_dimension)
        ff = PositionwiseFeedForward(model_dimension, feedforward_dimension, dropout)
        position = PositionalEncoding(model_dimension, dropout)

        self.encoder = Encoder(EncoderLayer(model_dimension, c(attn), c(ff), dropout), num_layers)
        self.decoder = Decoder(
            DecoderLayer(model_dimension, c(attn), c(attn), c(ff), dropout), num_layers
        )
        self.src_embed = nn.Sequential(Embeddings(model_dimension, vocabulary_size), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(model_dimension, vocabulary_size), c(position))
        self.generator = Generator(model_dimension, vocabulary_size)

        self._init_params()

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        Initialize parameters with Glorot / fan_avg.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_params(self):
        # parameter_enums = ModelParametersEnum
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return dict(
            vocabulary_size=self.vocabulary_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            model_dimension=self.model_dimension,
            feedforward_dimension=self.feedforward_dimension,
            dropout=self.dropout,
        )
