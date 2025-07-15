"""The Pepinvent sampling module"""

__all__ = ["PepinventSampler"]
from typing import List, Tuple
import logging

import torch.utils.data as tud
from rdkit import Chem
from reinvent.chemistry import tokens

from .sampler import Sampler, validate_smiles
from . import params
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.models.model_factory.sample_batch import SampleBatch, BatchRow
from reinvent.chemistry.tokens import PEPINVENT_CHUCKLES_SEPARATOR_TOKEN, PEPINVENT_MASK_TOKEN

logger = logging.getLogger(__name__)


class PepinventSampler(Sampler):
    """Carry out sampling with Pepinvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the Pepinvent model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: SampleBatch
        """
        if self.sample_strategy == "multinomial":
            smilies = smilies * self.batch_size

        tokenizer = SMILESTokenizer()
        dataset = Dataset(smilies, self.model.get_vocabulary(), tokenizer)
        dataloader = tud.DataLoader(
            dataset,
            batch_size=params.DATALOADER_BATCHSIZE,
            shuffle=False,
            collate_fn=Dataset.collate_fn,
        )

        sequences = []

        for batch in dataloader:
            src, src_mask = batch

            sampled = self.model.sample(src, src_mask, self.sample_strategy)

            for batch_row in sampled:
                sequences.append(batch_row)

        sampled = SampleBatch.from_list(sequences)

        mols, sampled = self._join_fragments(sampled)

        sampled.smilies, sampled.states = validate_smiles(
            mols, sampled.smilies, isomeric=self.isomeric, return_original_smiles=True
        )

        return sampled

    def _join_fragments(self, sequences: SampleBatch) -> Tuple[List[Chem.Mol], SampleBatch]:
        """Join input masked peptide with generated fillers

        :param sequences: a batch of sequences
        :returns: a list of RDKit molecules and SampleBatch where smilies field is joined complete smiles
        """

        mols = []
        samples = []
        for sample in sequences:
            smiles = sample.input
            num_fillers = sample.output.count("|") + 1
            num_mask = smiles.count("?")
            # The number of fillers generated is less than the number of masked slots
            if num_fillers < num_mask:
                mol = None
            # The number of fillers generated is greater than the number of masked slots
            elif num_fillers > num_mask:
                # Ignore the extra generated amino acids
                sample.output = "|".join(
                    sample.output.split(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN)[:num_mask]
                )
                mol, complete_smiles = self._create_complete_mol(sample)
                sample.smiles = complete_smiles
            elif num_fillers == num_mask:
                mol, complete_smiles = self._create_complete_mol(sample)
                sample.smiles = complete_smiles

            mols.append(mol)
            samples.append(sample)
        sampled = SampleBatch.from_list(samples)

        return mols, sampled

    def _create_complete_mol(self, sample: BatchRow) -> Tuple[Chem.Mol, str]:
        smiles = sample.input
        # Put filler in the masked position
        for replacement in sample.output.split(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN):
            smiles = smiles.replace(tokens.PEPINVENT_MASK_TOKEN, replacement, 1)
        # replace the chuckles separator token with an empty string
        complete_smiles = smiles.replace(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN, "")

        return Chem.MolFromSmiles(complete_smiles), complete_smiles

    @classmethod
    def split_fillers(cls, sampled: SampleBatch) -> Tuple[List[str], List[str]]:
        # example split_fillers [['C', 'O'], ['C', 'CC', 'CCC'], ['N']]
        split_fillers = [
            filler.split(PEPINVENT_CHUCKLES_SEPARATOR_TOKEN) for filler in sampled.items2
        ]
        # Number of fillers to log = number of masked spots
        num_fillers_to_save = sampled.items1[0].count(PEPINVENT_MASK_TOKEN)
        # Create headers
        filler_headers = [f"Filler_{i + 1}" for i in range(num_fillers_to_save)]

        # Create Filler columns
        # when number of fillers > num_fillers_to_save, discard the rest;
        # when the number of fillers < num_fillers_to_save, fill with None
        # e.g. [('C', 'C', 'N'), ('O', 'CC', None)]
        filler_columns = list(
            zip(
                *[
                    fillers[:num_fillers_to_save] + [None] * (num_fillers_to_save - len(fillers))
                    for fillers in split_fillers
                ]
            )
        )

        return filler_headers, filler_columns
