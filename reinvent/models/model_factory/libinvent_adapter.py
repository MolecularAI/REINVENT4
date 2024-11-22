"""Adapter for LibInvent"""

from __future__ import annotations

__all__ = ["LibinventAdapter", "LibinventTransformerAdapter"]
from typing import List, TYPE_CHECKING

import torch

from .sample_batch import SampleBatch
from reinvent.models.libinvent.models.dataset import DecoratorDataset
from reinvent.models.model_factory.model_adapter import (
    ModelAdapter,
    SampledSequencesDTO,
    BatchLikelihoodDTO,
)
from reinvent.models.model_factory.transformer_adapter import TransformerAdapter

if TYPE_CHECKING:
    pass


class LibinventAdapter(ModelAdapter):
    def likelihood(
        self,
        scaffold_seqs: torch.Tensor,
        scaffold_seq_lengths: torch.Tensor,
        decoration_seqs: torch.Tensor,
        decoration_seq_lengths: torch.Tensor,
    ):
        return self.model.likelihood(
            scaffold_seqs, scaffold_seq_lengths, decoration_seqs, decoration_seq_lengths
        )

    def likelihood_smiles(
        self, sampled_sequence_list: List[SampledSequencesDTO]
    ) -> BatchLikelihoodDTO:
        return self.likelihood_smiles_common(DecoratorDataset, sampled_sequence_list)

    def sample(self, scaffold_seqs, scaffold_seq_lengths) -> SampleBatch:
        # scaffold SMILES, decoration SMILES, NLLs
        sampled = self.model.sample_decorations(scaffold_seqs, scaffold_seq_lengths)
        return SampleBatch(*sampled)


class LibinventTransformerAdapter(TransformerAdapter):
    pass
