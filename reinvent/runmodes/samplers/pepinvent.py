"""The Pepinvent sampling module"""

__all__ = ["PepinventSampler"]
from typing import List
import logging

import torch.utils.data as tud
from rdkit import Chem
from reinvent.chemistry import tokens

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.models.model_factory.sample_batch import SampleBatch

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

        mols = self._join_fragments(sampled)

        sampled.smilies, sampled.states = validate_smiles(
            mols, sampled.output, isomeric=self.isomeric
        )

        return sampled

    def _join_fragments(self, sequences: SampleBatch) -> List[Chem.Mol]:
        """Join input masked peptide with generated fillers

        :param sequences: a batch of sequences
        :returns: a list of RDKit molecules
        """

        mols = []

        for sample in sequences:
            smiles = sample.input
            for replacement in sample.output.split(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN):
                smiles = smiles.replace(tokens.PEPINVENT_MASK_TOKEN, replacement, 1)
            mol = Chem.MolFromSmiles(smiles.replace(tokens.PEPINVENT_CHUCKLES_SEPARATOR_TOKEN, ""))
            mols.append(mol)

        return mols
