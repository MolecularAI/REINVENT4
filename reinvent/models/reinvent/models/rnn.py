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
        device=torch.device("cpu"),
    ) -> None:
        """N layer RNN cell including embedding layer and output linear layer

        Supports both GRU and LSTM modes.

        :param voc_size: size of the vocabulary
        :param layer_size: size of each of the RNN layers
        :param num_layers: number of RNN layers
        :param cell_type: RNN mode, either "gru" or "lstm"
        :param embedding_layer_size: size of the embedding layer
        :param dropout: dropout probability
        :param layer_normalization: shpuöd layer should be normalized or not
        :param device: the PyTorch device
        :raises RuntimeError: if cell_type is invalid
        """

        super(RNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization
        self.device = device

        self._embedding = tnn.Embedding(voc_size, self._embedding_layer_size)

        rnn = getattr(tnn, self._cell_type.upper(), None)

        if not rnn:
            raise RuntimeError('cell type must be either "gru" or "lstm"')

        self._rnn: tnn.RNNBase = rnn(
            input_size=self._embedding_layer_size,
            hidden_size=self._layer_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
            batch_first=True,
            device=self.device,
        )

        self._linear = tnn.Linear(self._layer_size, voc_size)

    def forward(
        self,
        input_vector: torch.Tensor,
        hidden_state: torch.Tensor | Sequence[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on the RNN

        Note: you need to pass the **whole** sequence.

        :param input_vector: input tensor (batch_size, seq_size)
        :param hidden_state: hidden state tensor (optional)
        :raises ValueError: raised when cell type is unknown
        :returns: output, hidden state
        """

        batch_size, seq_size = input_vector.size()

        if hidden_state is None:
            size = (self._num_layers, batch_size, self._layer_size)

            if self._cell_type == "gru":
                hidden_state = torch.zeros(*size, device=self.device)
            elif self._cell_type == "lstm":
                hidden_state = [
                    torch.zeros(*size, device=self.device),
                    torch.zeros(*size, device=self.device),
                ]
            else:
                raise ValueError(f'Invalid cell type "{self._cell_type}"')

        embedded_data = self._embedding(input_vector)  # (batch,seq,embedding)
        output_vector, hidden_state_out = self._rnn(embedded_data, hidden_state)

        if self._layer_normalization:
            output_vector = tnnf.layer_norm(output_vector, output_vector.size()[1:])

        output_vector = output_vector.reshape(-1, self._layer_size)
        output_data = self._linear(output_vector).view(batch_size, seq_size, -1)

        return output_data, hidden_state_out

    def get_params(self) -> Dict[str, Any]:
        """Get configuration parameters of the RNN

        :returns: the RNN's parameters
        """

        return {
            "dropout": self._dropout,
            "layer_size": self._layer_size,
            "num_layers": self._num_layers,
            "cell_type": self._cell_type,
            "embedding_layer_size": self._embedding_layer_size,
            "layer_normalization": self._layer_normalization,
        }
