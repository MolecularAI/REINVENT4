"""The LibInvent sampling module"""

__all__ = ["LibinventSampler", "LibinventTransformerSampler"]
from typing import List
import logging

import torch.utils.data as tud
from rdkit import Chem

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.libinvent.models.dataset import Dataset
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.chemistry import conversions
from reinvent.chemistry.library_design import attachment_points, bond_maker
from reinvent.models.transformer.core.dataset.dataset import Dataset as TransformerDataset

logger = logging.getLogger(__name__)


class LibinventSampler(Sampler):
    """Carry out sampling with LibInvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the LibInvent model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: SampleBatch
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

        mols = self._join_fragments(sampled)

        sampled.smilies, sampled.states = validate_smiles(
            mols, sampled.output, isomeric=self.isomeric
        )

        return sampled

    def _standardize_input(self, scaffold_list: List[str]):
        return [conversions.convert_to_standardized_smiles(scaffold) for scaffold in scaffold_list]

    def _get_randomized_smiles(self, scaffolds: List[str]):
        """Randomize the scaffold SMILES"""

        scaffold_mols = [conversions.smile_to_mol(scaffold) for scaffold in scaffolds]
        randomized = [bond_maker.randomize_scaffold(mol) for mol in scaffold_mols]

        return randomized

    def _join_fragments(self, sequences: SampleBatch) -> List[Chem.Mol]:
        """Join input scaffold and generated decorators

        :param sequences: a batch of sequences
        :returns: a list of RDKit molecules
        """

        mols = []

        for sample in sequences:
            input_scaffold = sample.input
            decorators = sample.output

            scaffold = attachment_points.add_attachment_point_numbers(
                input_scaffold, canonicalize=False
            )
            mol: Chem.Mol = bond_maker.join_scaffolds_and_decorations(  # may return None
                scaffold, decorators, keep_labels_on_atoms=True
            )

            mols.append(mol)

        return mols


LibinventTransformerSampler = LibinventSampler
