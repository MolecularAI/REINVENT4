"""Adapter for Mol2Mol"""

from __future__ import annotations

__all__ = ["TransformerAdapter"]

from abc import ABC
from typing import List, Tuple

import torch
import torch.utils.data as tud

from .sample_batch import SampleBatch
from reinvent.models.model_factory.model_adapter import (
    ModelAdapter,
    SampledSequencesDTO,
    BatchLikelihoodDTO,
)
from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum

# Internal batch size for computing the likelihood
LIKELIHOOD_BATCH_SIZE = 32


class TransformerAdapter(ModelAdapter, ABC):
    def likelihood(self, src, src_mask, trg, trg_mask) -> torch.Tensor:
        return self.model.likelihood(src, src_mask, trg, trg_mask)

    def likelihood_smiles(
        self, sampled_sequence_list: List[SampledSequencesDTO]
    ) -> BatchLikelihoodDTO:
        input = [dto.input for dto in sampled_sequence_list]
        output = [dto.output for dto in sampled_sequence_list]
        dataset = PairedDataset(input, output, vocabulary=self.vocabulary, tokenizer=self.tokenizer)
        data_loader = tud.DataLoader(
            dataset,
            LIKELIHOOD_BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            collate_fn=PairedDataset.collate_fn,
        )
        likelihood = []

        for batch in data_loader:
            likelihood.append(
                self.likelihood(batch.input, batch.input_mask, batch.output, batch.output_mask)
            )
        dto = BatchLikelihoodDTO(torch.cat(likelihood, 0))
        return dto

    def sample(self, src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL) -> Tuple:
        # input SMILES, output SMILES, NLLs
        sampled = self.model.sample(src, src_mask, decode_type)
        return SampleBatch(*sampled)

    def set_beam_size(self, beam_size: int):
        self.model.set_beam_size(beam_size)

    def set_temperature(self, temperature: float = 1.0):
        self.model.set_temperature(temperature)
