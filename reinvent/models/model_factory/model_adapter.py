"""Class to provide a (mostly) unified interface to all the models

The base class holds code that is generic to the models.
"""

from __future__ import annotations

__all__ = ["ModelAdapter", "SampledSequencesDTO", "BatchLikelihoodDTO"]
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
import torch.utils.data as tud


logger = logging.getLogger(__name__)


@dataclass
class SampledSequencesDTO:
    # Libinvent, Linkinvent, Mol2Mol

    input: str  # SMILES
    output: str  # SMILES
    nll: float  # negative log likelihood


@dataclass
class LinkInventBatchDTO:
    input: torch.Tensor
    output: torch.Tensor


@dataclass
class BatchLikelihoodDTO:
    likelihood: torch.Tensor


class ModelAdapter(ABC):
    def __init__(self, model):
        self.model = model
        self.model_type = model._model_type
        self.version = model._version
        self.vocabulary = model.vocabulary
        self.max_sequence_length = model.max_sequence_length
        self.network = model.network
        self.device = model.device

        # FIXME: ugly hard-coded list
        for name in ["tokenizer"]:
            if hasattr(model, name):
                attr = getattr(model, name)
                setattr(self, name, attr)

    @abstractmethod
    def likelihood(self, *args, **kwargs):
        """Compote NLL from token sequences"""

    @abstractmethod
    def likelihood_smiles(self, *args, **kwargs):
        """Compute NLL from SMILES"""

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Get a sample from the model"""

    def likelihood_smiles_common(self, Dataset, sequences: List[SampledSequencesDTO]):
        """Calculates the NLL for a set of SMILES strings

        Common to both LibInvent and LinkInvent.

        :param sequences: list with pairs of (scaffold, decoration) SMILES.
        :param Dataset: dataset class specific to LibInvent and LinkInvent
        :return: a tuple that follows the same order as the input list of SampledSequencesDTO.
        """

        sequence_pairs = [[ss.input, ss.output] for ss in sequences]
        dataset = Dataset(sequence_pairs, self.get_vocabulary())

        dataloader = tud.DataLoader(
            dataset,
            batch_size=len(dataset),
            collate_fn=Dataset.collate_fn,
            shuffle=False,
            generator=torch.Generator(device=self.device),
        )

        for _input, _output in dataloader:
            nlls = self.likelihood(*_input, *_output)
            dto = BatchLikelihoodDTO(nlls)

            return dto

    def save_to_file(self, path) -> None:
        """Save model to a pickle file

        :param path: path for the pickle file
        """

        self.model.save(path)

    def get_save_dict(self):
        return self.model.get_save_dict()

    def set_mode(self, mode: str) -> None:
        """Set training or evaluation mode.

        FIXME: use bool for mode as in torch

        :param mode: "training" or "inference"
        """

        if mode == "training":
            self.network.train()
        elif mode == "inference":
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    def get_network_parameters(self):
        """Get the parameters for the network"""

        return self.model.get_network_parameters()

    def get_vocabulary(self):
        """Get the vocabulary for the model"""
        # FIXME all models return different data structures
        # for lib and linkinvent, model.vocabulary is a combination of two vocabularies and their tokenizers
        # linkinvent -> PairedModelVocabulary
        # libinvent -> DecoratorVocabulary
        # reinvent -> Vocabulary
        return self.vocabulary

    def set_max_sequence_length(self, max_sequence_length: int) -> None:
        """Set the maximum sequence length for the model"""
        if max_sequence_length is None:
            return
        if max_sequence_length < 5:
            return
        self.max_sequence_length = max_sequence_length

        # FIXME: only available in Mol2Mol
        self.model.max_sequence_length = max_sequence_length
