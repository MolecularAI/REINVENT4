import torch
from torch import nn as tnn
from torch.nn.utils import rnn as tnnur

from reinvent.models.model_parameter_enum import ModelParametersEnum


class Encoder(tnn.Module):
    """
    Simple bidirectional RNN encoder implementation.
    """

    def __init__(self, num_layers: int, num_dimensions: int, vocabulary_size: int, dropout: float):
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

    def forward(
        self, padded_seqs: torch.Tensor, seq_lengths: torch.Tensor
    ) -> (torch.Tensor, (torch.Tensor, torch.Tensor),):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the sequences (batch, seq).
        :param seq_lengths: The lengths of the sequences (for packed sequences).
        :return : A tensor with all the output values for each step and the two hidden states.
        """
        batch_size, max_seq_size = padded_seqs.size()
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
        hs_h = hs_h.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(
            dim=1
        )  # (layers, batch, dim)
        hs_c = hs_c.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(
            dim=1
        )  # (layers, batch, dim)
        padded_seqs = padded_seqs.view(batch_size, max_seq_size, 2, self.num_dimensions).sum(
            dim=2
        )  # (batch, seq, dim)

        return padded_seqs, (hs_h, hs_c)

    def _initialize_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(self.num_layers * 2, batch_size, self.num_dimensions)

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
