"""
Implementation of the decorator using a Encoder-Decoder architecture.
"""

import math
import logging

import torch
import torch.nn as tnn
import torch.nn.utils.rnn as tnnur

from reinvent.models.model_parameter_enum import ModelParametersEnum

logger = logging.getLogger(__name__)


class Encoder(tnn.Module):
    """
    Simple bidirectional RNN encoder implementation.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
        super(Encoder, self).__init__()

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
            bidirectional=True,
        )

    def forward(self, padded_seqs, seq_lengths):  # pylint: disable=arguments-differ
        # FIXME: This fails with a batch of 1 because squeezing looses a dimension with size 1
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the sequences (batch, seq).
        :param seq_lengths: The lengths of the sequences (for packed sequences).
        :return : A tensor with all the output values for each step and the two hidden states.
        """
        batch_size = padded_seqs.size(0)
        max_seq_size = padded_seqs.size(1)
        hidden_state = self._initialize_hidden_state(batch_size)

        padded_seqs = self._embedding(padded_seqs)
        hs_h, hs_c = (hidden_state, hidden_state.clone().detach())

        # FIXME: this is to guard against non compatible `gpu` input for pack_padded_sequence() method in pytorch 1.7
        seq_lengths = seq_lengths.cpu()

        packed_seqs = tnnur.pack_padded_sequence(
            padded_seqs, seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_seqs, (hs_h, hs_c) = self._rnn(packed_seqs, (hs_h, hs_c))
        padded_seqs, _ = tnnur.pad_packed_sequence(packed_seqs, batch_first=True)

        # sum up bidirectional layers and collapse
        hs_h = (
            hs_h.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1).squeeze()
        )  # (layers, batch, dim)
        hs_c = (
            hs_c.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1).squeeze()
        )  # (layers, batch, dim)
        padded_seqs = (
            padded_seqs.view(batch_size, max_seq_size, 2, self.num_dimensions).sum(dim=2).squeeze()
        )  # (batch, seq, dim)

        return padded_seqs, (hs_h, hs_c)

    def _initialize_hidden_state(self, batch_size):
        return torch.zeros(self.num_layers * 2, batch_size, self.num_dimensions)

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """

        parameter_enums = ModelParametersEnum

        return {
            parameter_enums.NUMBER_OF_LAYERS: self.num_layers,
            parameter_enums.NUMBER_OF_DIMENSIONS: self.num_dimensions,
            parameter_enums.VOCABULARY_SIZE: self.vocabulary_size,
            parameter_enums.DROPOUT: self.dropout,
        }


class AttentionLayer(tnn.Module):
    def __init__(self, num_dimensions):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions

        self._attention_linear = tnn.Sequential(
            tnn.Linear(self.num_dimensions * 2, self.num_dimensions), tnn.Tanh()
        )

    def forward(
        self, padded_seqs, encoder_padded_seqs, decoder_mask
    ):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_mask: A tensor that represents the encoded input mask.
        :return : Two tensors: one with the modified logits and another with the attention weights.
        """
        # scaled dot-product
        # (batch, seq_d, 1, dim)*(batch, 1, seq_e, dim) => (batch, seq_d, seq_e*)
        attention_weights = (
            (padded_seqs.unsqueeze(dim=2) * encoder_padded_seqs.unsqueeze(dim=1))
            .sum(dim=3)
            .div(math.sqrt(self.num_dimensions))
            .softmax(dim=2)
        )
        # (batch, seq_d, seq_e*)@(batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        return (
            self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2))
            * decoder_mask,
            attention_weights,
        )


class Decoder(tnn.Module):
    """
    Simple RNN decoder.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
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
        self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states: tuple
    ):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param seq_lengths: A list with the length of each output sequence.
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
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

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """

        parameter_enum = ModelParametersEnum

        return {
            parameter_enum.NUMBER_OF_LAYERS: self.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self.num_dimensions,
            parameter_enum.VOCABULARY_SIZE: self.vocabulary_size,
            parameter_enum.DROPOUT: self.dropout,
        }


class Decorator(tnn.Module):
    """
    An encoder-decoder that decorates scaffolds.
    """

    def __init__(self, encoder_params, decoder_params):
        super(Decorator, self).__init__()

        self._encoder = Encoder(**encoder_params)
        self._decoder = Decoder(**decoder_params)

    def forward(
        self, encoder_seqs, encoder_seq_lengths, decoder_seqs, decoder_seq_lengths
    ):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param encoder_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_seq_lengths: A list with the length of each input sequence.
        :param decoder_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_seq_lengths: The lengths of the decoder sequences.
        :return : The output logits as a tensor (batch, seq_d, dim).
        """
        encoder_padded_seqs, hidden_states = self.forward_encoder(encoder_seqs, encoder_seq_lengths)
        logits, _, _ = self.forward_decoder(
            decoder_seqs, decoder_seq_lengths, encoder_padded_seqs, hidden_states
        )
        return logits

    def forward_encoder(self, padded_seqs, seq_lengths):
        """
        Does a forward pass only of the encoder.
        :param padded_seqs: The data to feed the encoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns a tuple with (encoded_seqs, hidden_states)
        """
        return self._encoder(padded_seqs, seq_lengths)

    def forward_decoder(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):
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
