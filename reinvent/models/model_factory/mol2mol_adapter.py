"""Adapter for Mol2Mol"""

from __future__ import annotations

__all__ = ["Mol2MolAdapter"]
from typing import List, Tuple

import torch
import torch.utils.data as tud
import numpy as np

from .sample_batch import SampleBatch
from reinvent.models.mol2mol.dataset.paired_dataset import PairedDataset
from reinvent.models.mol2mol.enums import SamplingModesEnum
from reinvent.models.model_factory.model_adapter import (
    ModelAdapter,
    SampledSequencesDTO,
    BatchLikelihoodDTO,
)


class Mol2MolAdapter(ModelAdapter):
    def likelihood(self, src, src_mask, trg, trg_mask) -> torch.Tensor:
        return self.model.likelihood(src, src_mask, trg, trg_mask)

    def likelihood_smiles(
        self, sampled_sequence_list: List[SampledSequencesDTO]
    ) -> BatchLikelihoodDTO:
        input = [dto.input for dto in sampled_sequence_list]
        output = [dto.output for dto in sampled_sequence_list]
        dataset = PairedDataset(input, output, vocabulary=self.vocabulary, tokenizer=self.tokenizer)
        data_loader = tud.DataLoader(
            dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn
        )

        for batch in data_loader:
            likelihood = self.likelihood(
                batch.input, batch.input_mask, batch.output, batch.output_mask
            )
            dto = BatchLikelihoodDTO(batch, likelihood)
            return dto

    def sample(self, src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL) -> Tuple:
        # input SMILES, output SMILES, NLLs
        sampled = self.model.sample(src, src_mask, decode_type)
        return SampleBatch(*sampled)

    def sample_smiles(self, dataloader, num: int = 128, batch_size: int = 128):
        # batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]

        for batch in dataloader:
            src, src_mask, _, _, _ = batch

            # for some reason those two end up on the CPU
            src = src.to(self.device)
            src_mask = src_mask.to(self.device)

            # FIXME: decode type
            _, smilies, nlls = self.model.sample(src, src_mask, SamplingModesEnum.MULTINOMIAL)

            return smilies, np.array(nlls)

    # NOTE: unique to Mol2Mol
    def set_beam_size(self, beam_size: int):
        self.model.set_beam_size(beam_size)

    def set_temperature(self, temperature: float = 1.0):
        self.model.set_temperature(temperature)
