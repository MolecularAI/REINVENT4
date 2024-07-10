"""Reinvent adapter"""

__all__ = ["ReinventAdapter"]
from typing import List

import torch

from .sample_batch import SampleBatch
from reinvent.models.model_factory.model_adapter import ModelAdapter


class ReinventAdapter(ModelAdapter):
    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.model.likelihood(sequences)

    def likelihood_smiles(self, smiles: List[str]) -> torch.Tensor:
        return self.model.likelihood_smiles(smiles)

    def sample(self, batch_size: int) -> SampleBatch:
        """Sample from Model

        :param batch_size: batch size
        :returns: token sequences, list of SMILES, NLLs
        """
        # torch.Tensor, List[str], torch.Tensor
        sequences, smilies, nlls = self.model.sample(batch_size)

        # NOTE: keep the sequences and nlls as Tensor as they are needed for
        #       later computations
        return SampleBatch(sequences, smilies, nlls)
