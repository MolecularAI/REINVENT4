"""
Implementation of a network using an Encoder-Decoder architecture.
"""

import torch.nn as tnn
from torch import Tensor

from reinvent.models.linkinvent.networks.decoder import Decoder
from reinvent.models.linkinvent.networks.encoder import Encoder


class EncoderDecoder(tnn.Module):
    """
    An encoder-decoder that combines input with generated targets.
    """

    def __init__(self, encoder_params: dict, decoder_params: dict):
        super(EncoderDecoder, self).__init__()

        self._encoder = Encoder(**encoder_params)
        self._decoder = Decoder(**decoder_params)

    def forward(
        self,
        encoder_seqs: Tensor,
        encoder_seq_lengths: Tensor,
        decoder_seqs: Tensor,
        decoder_seq_lengths: Tensor,
    ):
        """
        Performs the forward pass.
        :param encoder_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_seq_lengths: A list with the length of each input sequence.
        :param decoder_seqs: A tensor with the encoded input input sequences (batch, seq_e, dim).
        :param decoder_seq_lengths: The lengths of the decoder sequences.
        :return : The output logits as a tensor (batch, seq_d, dim).
        """
        encoder_padded_seqs, hidden_states = self.forward_encoder(encoder_seqs, encoder_seq_lengths)
        logits, _, _ = self.forward_decoder(
            decoder_seqs, decoder_seq_lengths, encoder_padded_seqs, hidden_states
        )
        return logits

    def forward_encoder(self, padded_seqs: Tensor, seq_lengths: Tensor):
        """
        Does a forward pass only of the encoder.
        :param padded_seqs: The data to feed the encoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns a tuple with (encoded_seqs, hidden_states)
        """
        return self._encoder(padded_seqs, seq_lengths)

    def forward_decoder(
        self,
        padded_seqs: Tensor,
        seq_lengths: Tensor,
        encoder_padded_seqs: Tensor,
        hidden_states: Tensor,
    ):
        """
        Does a forward pass only of the decoder.
        :param hidden_states: The hidden states from the encoder.
        :param padded_seqs: The data to feed to the decoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns the logits and the hidden state for each element of the sequence passed.
        """
        return self._decoder(padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states)

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "encoder_params": self._encoder.get_params(),
            "decoder_params": self._decoder.get_params(),
        }
