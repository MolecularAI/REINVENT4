"""The LibInvent sampling module"""

__all__ = ["LibinventSampler", "LibinventTransformerSampler"]
from typing import List, Tuple
import logging

import torch
import torch.utils.data as tud

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.libinvent.models.dataset import Dataset
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.runmodes.utils.helpers import join_fragments
from reinvent.chemistry import conversions
from reinvent.chemistry.library_design import attachment_points, bond_maker
from reinvent.models.transformer.core.dataset.dataset import Dataset as TransformerDataset

logger = logging.getLogger(__name__)


class LibinventSampler(Sampler):
    """Carry out sampling with LibInvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the LibInvent model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: list of SampledSequencesDTO
        """

        if self.model.version == 2:  # Transformer-based
            smilies = self._standardize_input(smilies)

        scaffolds = self._get_randomized_smiles(smilies) if self.randomize_smiles else smilies

        clean_scaffolds = [
            attachment_points.remove_attachment_point_numbers(scaffold)
            for scaffold in scaffolds
            if scaffold
        ]

        if self.model.version == 1:  # RNN-based
            clean_scaffolds = clean_scaffolds * self.batch_size

            dataset = Dataset(
                clean_scaffolds,
                self.model.get_vocabulary().scaffold_vocabulary,
                self.model.get_vocabulary().scaffold_tokenizer,
            )
        elif self.model.version == 2:  # Transformer-based
            if self.sample_strategy == "multinomial":
                clean_scaffolds = clean_scaffolds * self.batch_size

            dataset = TransformerDataset(
                clean_scaffolds, self.model.get_vocabulary(), self.model.tokenizer
            )

        dataloader = tud.DataLoader(
            dataset,
            batch_size=params.DATALOADER_BATCHSIZE,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        sequences = []

        for batch in dataloader:
            inputs, input_info = batch
            if self.model.version == 1:
                sampled = self.model.sample(inputs, input_info)
            elif self.model.version == 2:
                sampled = self.model.sample(inputs, input_info, self.sample_strategy)
            for batch_row in sampled:
                sequences.append(batch_row)
        sampled = SampleBatch.from_list(sequences)

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled)

        mols = join_fragments(sampled, reverse=False, keep_labels=True)

        sampled.smilies, sampled.states = validate_smiles(
            mols, sampled.output, isomeric=self.isomeric
        )

        return sampled

    def _standardize_input(self, scaffold_list: List[str]):
        return [conversions.convert_to_standardized_smiles(scaffold)
                    for scaffold in scaffold_list]

    def _get_randomized_smiles(self, scaffolds: List[str]):
        """Randomize the scaffold SMILES"""

        scaffold_mols = [conversions.smile_to_mol(scaffold) for scaffold in scaffolds]
        randomized = [bond_maker.randomize_scaffold(mol) for mol in scaffold_mols]

        return randomized


LibinventTransformerSampler = LibinventSampler
