"""LSTM (or GRU) RNN for the classical Reinvent de novo model"""

from typing import Tuple, Dict, Sequence, Any

import torch
import torch.nn as tnn
import torch.nn.functional as tnnf


class RNN(tnn.Module):
    """
    Implements an N layer GRU(M) or LSTM cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(
            self,
            voc_size: int,
            layer_size: int = 512,
            num_layers: int = 3,
            cell_type: str = "gru",
            embedding_layer_size: int = 256,
            dropout: float = 0.0,
            layer_normalization: bool = False,
    ) -> None:
        """
        Implements an N layer GRU|LSTM cell including an embedding layer and an output linear layer back
        to the size of the vocabulary.

        :param voc_size: Size of the vocabulary.
        :param layer_size: Size of each of the RNN layers.
        :param num_layers: Number of RNN layers.
        :param cell_type: Type of RNN: either gru or lstm.
        :param embedding_layer_size: Size of the embedding layer.
        :param dropout: Dropout probability.
        :param layer_normalization: Selects whether layer should be normalized or not.
        :raises ValueError: raised if cell_type is unknown
        """

        super(RNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization

        self._embedding = tnn.Embedding(voc_size, self._embedding_layer_size)

        self._rnn: tnn.RNNBase  # base class of GRU and LSTM

        if self._cell_type == "gru":
            self._rnn = tnn.GRU(
                self._embedding_layer_size,
                self._layer_size,
                num_layers=self._num_layers,
                dropout=self._dropout,
                batch_first=True,
            )
        elif self._cell_type == "lstm":
            self._rnn = tnn.LSTM(
                self._embedding_layer_size,
                self._layer_size,
                num_layers=self._num_layers,
                dropout=self._dropout,
                batch_first=True,
            )
        else:
            raise ValueError('Value of the parameter cell_type should be "gru" or "lstm"')

        self._linear = tnn.Linear(self._layer_size, voc_size)

    def forward(
            self,
            input_vector: torch.Tensor,
            hidden_state: torch.Tensor | Sequence[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.

        :param input_vector: input tensor (batch_size, seq_size)
        :param hidden_state: hidden state tensor (optional)
        :raises ValueError: raised when cell type is unknown
        :returns: output, hidden state
        """

        batch_size, seq_size = input_vector.size()

        if hidden_state is None:
            size = (self._num_layers, batch_size, self._layer_size)

            if self._cell_type == "gru":
                hidden_state = torch.zeros(*size)
            elif self._cell_type == "lstm":
                hidden_state = [torch.zeros(*size), torch.zeros(*size)]
            else:
                raise ValueError(f'Invalid cell type "{self._cell_type}"')

        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)  # recursive call?

        if self._layer_normalization:
            output_vector = tnnf.layer_norm(output_vector, output_vector.size()[1:])

        output_vector = output_vector.reshape(-1, self._layer_size)

        output_data = self._linear(output_vector).view(batch_size, seq_size, -1)

        # LSTM:
        #   output_data is 3D
        #   hidden_state_out is a tuple of two 3D tensors
        return output_data, hidden_state_out

    def get_params(self) -> Dict[str, Any]:
        """
        Returns the configuration parameters of the RNN model.

        :returns: the RNN's parameters
        """

        return {
            "dropout": self._dropout,
            "layer_size": self._layer_size,
            "num_layers": self._num_layers,
            "cell_type": self._cell_type,
            "embedding_layer_size": self._embedding_layer_size,
        }