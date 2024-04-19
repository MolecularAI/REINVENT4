"""The LibInvent sampling module"""

__all__ = ["LibinventSampler"]
from typing import List, Tuple
import logging

import torch
import torch.utils.data as tud

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.libinvent.models.dataset import Dataset
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.runmodes.utils.helpers import join_fragments


logger = logging.getLogger(__name__)


class LibinventSampler(Sampler):
    """Carry out sampling with LibInvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the LibInvent model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: list of SampledSequencesDTO
        """

        scaffolds = self._get_randomized_smiles(smilies) if self.randomize_smiles else smilies

        clean_scaffolds = [
            self.chemistry.attachment_points.remove_attachment_point_numbers(scaffold)
            for scaffold in scaffolds
        ]

        # NOTE: for some reason there must be at least 2 scaffolds so need
        #       to "fake" a second one if only one given
        if len(clean_scaffolds) == 1:
            clean_scaffolds *= 2

        # FIXME: check why we need to amplify the dataset
        clean_scaffolds = clean_scaffolds * self.batch_size
        dataset = Dataset(
            clean_scaffolds,
            self.model.get_vocabulary().scaffold_vocabulary,
            self.model.get_vocabulary().scaffold_tokenizer,
        )

        dataloader = tud.DataLoader(
            dataset,
            batch_size=params.DATALOADER_BATCHSIZE,
            shuffle=False,
            collate_fn=Dataset.collate_fn,
        )

        sequences = []
        batch: Tuple[torch.Tensor, torch.Tensor]

        for batch in dataloader:
            scaffold_seqs, scaffold_seq_lengths = batch

            sampled = self.model.sample(scaffold_seqs, scaffold_seq_lengths)

            for batch_row in sampled:
                sequences.append(batch_row)

        sampled = SampleBatch.from_list(sequences)

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled)

        mols = join_fragments(sampled, reverse=False, keep_labels=True)

        sampled.smilies, sampled.states = validate_smiles(mols, sampled.output)

        return sampled

    def _get_randomized_smiles(self, scaffolds: List[str]):
        """Randomize the scaffold SMILES"""

        scaffold_mols = [
            self.chemistry.conversions.smile_to_mol(scaffold) for scaffold in scaffolds
        ]
        randomized = [self.chemistry.bond_maker.randomize_scaffold(mol) for mol in scaffold_mols]

        return randomized
