"""Adapter for LinkInvent"""

from __future__ import annotations

__all__ = ["LinkinventAdapter", "LinkinventTransformerAdapter"]
from typing import List, TYPE_CHECKING

from .sample_batch import SampleBatch
from reinvent.models.linkinvent.dataset.paired_dataset import PairedDataset
from reinvent.models.model_factory.model_adapter import (
    ModelAdapter,
    SampledSequencesDTO,
    BatchLikelihoodDTO,
)
from reinvent.models.model_factory.transformer_adapter import TransformerAdapter

if TYPE_CHECKING:
    pass


class LinkinventAdapter(ModelAdapter):
    def likelihood(self, warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths):
        return self.model.likelihood(
            warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths
        )

    def likelihood_smiles(
        self, sampled_sequence_list: List[SampledSequencesDTO]
    ) -> BatchLikelihoodDTO:
        return self.likelihood_smiles_common(PairedDataset, sampled_sequence_list)

    def sample(self, warheads_seqs, warheads_seq_lengths) -> SampleBatch:
        # warhead SMILES, linker SMILES, NLLs
        sampled = self.model.sample(warheads_seqs, warheads_seq_lengths)
        return SampleBatch(*sampled)


class LinkinventTransformerAdapter(TransformerAdapter):
    pass
