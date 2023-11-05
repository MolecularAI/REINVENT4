"""
Implementation of the RNN model
"""


import numpy as np
from typing import List, Tuple, Dict, Any, TypeVar, Sequence, Union, Iterator

import torch
import torch.nn as tnn
import torch.nn.functional as tnnf

from reinvent.models.reinvent.models import vocabulary as mv
from reinvent.models.reinvent.utils import collate_fn
from reinvent.models.model_mode_enum import ModelModeEnum


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
        hidden_state: Union[torch.Tensor, Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # pylint: disable=W0221
        """
        Performs a forward pass on the model. Note: you pass the **whole** sequence.

        :param input_vector: Input tensor (batch_size, seq_size).
        :param hidden_state: Hidden state tensor (optional).
        :raises ValueError: raised when cell type is unknwon
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

        return output_data, hidden_state_out

    def get_params(self) -> Dict[str, Any]:
        """
        Returns the configuration parameters of the RNN model.

        :returns: The RNN's parameters.
        """

        return {
            "dropout": self._dropout,
            "layer_size": self._layer_size,
            "num_layers": self._num_layers,
            "cell_type": self._cell_type,
            "embedding_layer_size": self._embedding_layer_size,
        }


M = TypeVar("M", bound="Model")


class Model:
    """
    Implements an RNN model using SMILES.
    """

    _model_type = "Reinvent"
    _version = 1

    def __init__(
        self,
        vocabulary: mv.Vocabulary,
        tokenizer: mv.SMILESTokenizer,
        network_params: dict = None,
        max_sequence_length: int = 256,
        mode: str = "training",
        device=torch.device("cpu"),
    ):
        """
        Implements an RNN using either GRU or LSTM.

        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the RNN class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        self._model_modes = ModelModeEnum()
        self.network = RNN(len(self.vocabulary), **network_params)
        self.network.to(device)
        self.device = device
        self.set_mode(mode)

        self._nll_loss = tnn.NLLLoss(reduction="none")

    def set_mode(self, mode: str) -> None:
        """
        Set training or inference mode of the network.

        :param mode: Mode to be set.
        :raises ValueError: raised when unknown mode
        """

        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    # hinting with M may require Python >= 3.9
    # def load_from_file(cls: type[M], file_path: str, sampling_mode: bool = False) -> M:
    @classmethod
    def load_from_file(cls, file_path: str, mode: str, device: torch.device):
        """
        Loads a model from a single file.

        :param file_path: input file file_path containing the model
        :param mode: whether to use sampling (inference, evaluation) or training mode
        :return: new instance of the RNN
        """

        save_dict = torch.load(file_path, map_location=device)

        return cls.create_from_dict(save_dict, mode, device)

    @classmethod
    def create_from_dict(cls, save_dict: dict, mode: str, device: torch.device):
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        network_params = save_dict.get("network_params", {})

        model = cls(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", mv.SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict["max_sequence_length"],
            mode=mode,
            device=device,
        )

        model.network.load_state_dict(save_dict["network"])

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            vocabulary=self.vocabulary,
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            network=self.network.state_dict(),
            network_params=self.network.get_params(),
        )

        return save_dict

    def save(self, file_path: str) -> None:
        """
        Saves the model into a file.

        :param file_path: Path to the model file.
        """

        save_dict = self.get_save_dict()

        torch.save(save_dict, file_path)

    save_to_file = save  # alias for backwards compatibility

    def likelihood_smiles(self, smiles: str) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        padded_sequences = collate_fn(sequences)
        return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """

        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)

        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    # NOTE: needed for Reinvent TL
    def sample_smiles(self, num: int = 128, batch_size: int = 128) -> Tuple[List[str], np.ndarray]:
        """
        Samples n SMILES from the model.  Is this batched because of memory concerns?

        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return: A list with SMILES and a list of likelihoods.
        """

        batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        for size in batch_sizes:
            if not size:
                break
            _, smiles, likelihoods = self.sample_sequences_and_smiles(batch_size=size)

            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())

            del likelihoods

        return smiles_sampled, np.concatenate(likelihoods_sampled)

    def sample_sequences_and_smiles(
        self, batch_size: int = 128
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        seqs, likelihoods = self._sample(batch_size=batch_size)
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
        ]

        return seqs, smiles, likelihoods

    # @torch.no_grad()
    def _sample(self, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        start_token = torch.zeros(batch_size, dtype=torch.long)
        start_token[:] = self.vocabulary["^"]
        input_vector = start_token
        sequences = [self.vocabulary["^"] * torch.ones([batch_size, 1], dtype=torch.long)]
        # NOTE: The first token never gets added in the loop so the sequences are initialized with a start token
        hidden_state = None
        nlls = torch.zeros(batch_size)

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)
            input_vector = torch.multinomial(probabilities, 1).view(-1)
            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        concat_sequences = torch.cat(sequences, 1)

        return concat_sequences.data, nlls

    def get_network_parameters(self) -> Iterator[tnn.Parameter]:
        """
        Returns the configuration parameters of the network.

        :returns: The neteworkparameters.
        """

        return self.network.parameters()
