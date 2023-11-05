import math

import torch
from torch import nn as tnn


class AttentionLayer(tnn.Module):

    def __init__(self, num_dimensions: int):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions

        self._attention_linear = tnn.Sequential(
            tnn.Linear(self.num_dimensions*2, self.num_dimensions),
            tnn.Tanh()
        )

    def forward(self, padded_seqs: torch.Tensor, encoder_padded_seqs: torch.Tensor, decoder_mask: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_padded_seqs: A tensor with the encoded input sequences (batch, seq_e, dim).
        :param decoder_mask: A tensor that represents the encoded input mask.
        :return : Two tensors: one with the modified logits and another with the attention weights.
        """
        # scaled dot-product
        # (batch, seq_d, 1, dim)*(batch, 1, seq_e, dim) => (batch, seq_d, seq_e*)
        attention_weights = (padded_seqs.unsqueeze(dim=2)*encoder_padded_seqs.unsqueeze(dim=1))\
            .sum(dim=3).div(math.sqrt(self.num_dimensions))\
            .softmax(dim=2)
        # (batch, seq_d, seq_e*)@(batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)

        return (self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2))*decoder_mask,
                attention_weights)
