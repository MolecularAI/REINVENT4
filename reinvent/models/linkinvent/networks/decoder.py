from typing import Tuple

import torch
from torch import nn as tnn
from torch.nn.utils import rnn as tnnur

from reinvent.models.linkinvent.networks.attention_layer import AttentionLayer
from reinvent.models.model_parameter_enum import ModelParametersEnum


class Decoder(tnn.Module):
    """
    Simple RNN decoder.
    """

    def __init__(self, num_layers: int, num_dimensions: int, vocabulary_size: int, dropout: float):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = tnn.Sequential(
            tnn.Embedding(self.vocabulary_size, self.num_dimensions),
            tnn.Dropout(dropout),
        )
        self._rnn = tnn.LSTM(
            self.num_dimensions,
            self.num_dimensions,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )

        self._attention = AttentionLayer(self.num_dimensions)

        self._linear = tnn.Linear(self.num_dimensions, self.vocabulary_size)  # just to redimension

    def forward(
        self,
        padded_seqs: torch.Tensor,
        seq_lengths: torch.Tensor,
        encoder_padded_seqs: torch.Tensor,
        hidden_states: Tuple[torch.Tensor],
    ) -> (torch.Tensor, Tuple[torch.Tensor], torch.Tensor,):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param seq_lengths: A list with the length of each output sequence.
        :param encoder_padded_seqs: A tensor with the encoded input sequences (batch, seq_e, dim).
        :param hidden_states: The hidden states from the encoder.
        :return : Three tensors: The output logits, the hidden states of the decoder and the attention weights.
        """
        # FIXME: this is to guard against non compatible `gpu` input for pack_padded_sequence() method in pytorch 1.7
        seq_lengths = seq_lengths.cpu()

        padded_encoded_seqs = self._embedding(padded_seqs)
        packed_encoded_seqs = tnnur.pack_padded_sequence(
            padded_encoded_seqs, seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_encoded_seqs, hidden_states = self._rnn(packed_encoded_seqs, hidden_states)
        padded_encoded_seqs, _ = tnnur.pad_packed_sequence(
            packed_encoded_seqs, batch_first=True
        )  # (batch, seq, dim)

        mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(
            padded_encoded_seqs, encoder_padded_seqs, mask
        )
        logits = self._linear(attn_padded_encoded_seqs) * mask  # (batch, seq, voc_size)
        return logits, hidden_states, attention_weights

    def get_params(self) -> dict:
        """Obtains the params for the network.

        :returns: A dict with the params.
        """

        parameter_enum = ModelParametersEnum

        return {
            parameter_enum.NUMBER_OF_LAYERS: self.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self.num_dimensions,
            parameter_enum.VOCABULARY_SIZE: self.vocabulary_size,
            parameter_enum.DROPOUT: self.dropout,
        }
