"""Classical Reinvent de novo model

See:
https://doi.org/10.1186/s13321-017-0235-x (original publication)
https://doi.org/10.1021/acs.jcim.0c00915 (REINVENT 2.0)
"""

from typing import List, Tuple, TypeVar, Iterator

import numpy as np
import torch
import torch.nn as tnn

from reinvent.models import meta_data
from reinvent.models.reinvent.models import rnn, vocabulary as mv
from reinvent.models.reinvent.utils import collate_fn
from reinvent.models.model_mode_enum import ModelModeEnum


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
        meta_data: meta_data.ModelMetaData,
        network_params: dict = None,
        max_sequence_length: int = 256,
        mode: str = "training",
        device=torch.device("cpu"),
    ):
        """
        Implements an RNN using either GRU or LSTM.

        :param vocabulary: vocabulary to use
        :param tokenizer: tokenizer to use
        :param meta_data: model meta data
        :param network_params: parameters required to initialize the RNN
        :param max_sequence_length: maximum length of sequence that can be generated
        :param mode: either "training" or "inference"
        :param device: the PyTorch device
        """

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.meta_data = meta_data
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        self._model_modes = ModelModeEnum()
        self.network = rnn.RNN(len(self.vocabulary), **network_params)
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

    @classmethod
    def create_from_dict(cls: type[M], save_dict: dict, mode: str, device: torch.device) -> M:
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        vocabulary = None

        if isinstance(save_dict["vocabulary"], dict):
            vocabulary = mv.Vocabulary.load_from_dictionary(save_dict["vocabulary"])
        else:
            vocabulary = save_dict["vocabulary"]

        model = cls(
            vocabulary=vocabulary,
            tokenizer=save_dict.get("tokenizer", mv.SMILESTokenizer()),
            meta_data=save_dict.get("metadata"),
            network_params=save_dict.get("network_params"),
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
            metadata=self.meta_data,
            vocabulary=self.vocabulary.get_dictionary(),
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
        meta_data.update_model_data(save_dict)
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

            _, smiles, likelihoods = self.sample(batch_size=size)

            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())

            del likelihoods

        return smiles_sampled, np.concatenate(likelihoods_sampled)

    @torch.no_grad()
    def sample(self, batch_size: int = 128) -> Tuple[torch.Tensor, list, torch.Tensor]:
        seqs, likelihoods = self._sample(batch_size=batch_size)

        # FIXME: this is potentially unnecessary in some cases
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
        ]

        return seqs, smiles, likelihoods

    def _sample(self, batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a number of sequences from the RNN

        :param batch_size: batch size which is the number of sequences to sample
        :returns: sequences (2D) and associated NLLs (1D)
        """

        # NOTE: the first token never gets added in the loop so initialize with the start token
        sequences = [torch.full((batch_size, 1), self.vocabulary[mv.START_TOKEN], dtype=torch.long)]
        input_vector = torch.full((batch_size,), self.vocabulary[mv.START_TOKEN], dtype=torch.long)
        hidden_state = None
        nlls = torch.zeros(batch_size)

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D
            log_probs = logits.log_softmax(dim=1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D
            input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D
            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        concat_sequences = torch.cat(sequences, dim=1)

        return concat_sequences.data, nlls

    def get_network_parameters(self) -> Iterator[tnn.Parameter]:
        """
        Returns the configuration parameters of the network.

        :returns: network parameters of the RNN
        """

        return self.network.parameters()
