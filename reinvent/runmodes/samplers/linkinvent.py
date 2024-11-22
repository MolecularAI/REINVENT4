"""The LinkInvent sampling module"""

__all__ = ["LinkinventSampler", "LinkinventTransformerSampler"]
from typing import List
import logging

import torch.utils.data as tud
from rdkit import Chem

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.linkinvent.dataset.dataset import Dataset
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.chemistry import conversions, tokens
from reinvent.chemistry.library_design import attachment_points, bond_maker
from ...models.transformer.core.dataset.dataset import Dataset as TransformerDataset

logger = logging.getLogger(__name__)


class LinkinventSampler(Sampler):
    """Carry out sampling with LinkInvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: SampleBatch
        """

        if self.model.version == 2:  # Transformer-based
            smilies = self._standardize_input(smilies)

        warheads_list = self._get_randomized_smiles(smilies) if self.randomize_smiles else smilies

        clean_warheads = [
            attachment_points.remove_attachment_point_numbers(warheads)
            for warheads in warheads_list
            if warheads
        ]

        if self.model.version == 1:  # RNN-based
            clean_warheads = clean_warheads * self.batch_size

            dataset = Dataset(clean_warheads, self.model.get_vocabulary().input)
        elif self.model.version == 2:  # Transformer-based
            if self.sample_strategy == "multinomial":
                clean_warheads = clean_warheads * self.batch_size

            dataset = TransformerDataset(
                clean_warheads, self.model.get_vocabulary(), self.model.tokenizer
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

    def _standardize_input(self, warheads_list: List[str]):
        cano_warheads_list = []
        for warheads in warheads_list:
            cano_warheads = "|".join(
                [
                    conversions.convert_to_standardized_smiles(warhead)
                    for warhead in warheads.split("|")
                ]
            )
            cano_warheads_list.append(cano_warheads)
        return cano_warheads_list

    def _get_randomized_smiles(self, warhead_pair_list: List[str]):
        """Randomize the warhead SMILES"""

        randomized_warhead_pair_list = []

        for warhead_pair in warhead_pair_list:
            warhead_list = warhead_pair.split(tokens.ATTACHMENT_SEPARATOR_TOKEN)
            warhead_mol_list = [conversions.smile_to_mol(warhead) for warhead in warhead_list]
            warhead_randomized_list = [
                conversions.mol_to_random_smiles(mol, isomericSmiles=self.isomeric)
                for mol in warhead_mol_list
            ]
            # Note do not use self.self._bond_maker.randomize_scaffold, as it would add unwanted brackets to the
            # attachment points (which are not part of the warhead vocabulary)
            warhead_pair_randomized = tokens.ATTACHMENT_SEPARATOR_TOKEN.join(
                warhead_randomized_list
            )
            randomized_warhead_pair_list.append(warhead_pair_randomized)

        return randomized_warhead_pair_list

    def _join_fragments(self, sequences: SampleBatch) -> List[Chem.Mol]:
        """Join input warheads with generated linker

        :param sequences: a batch of sequences
        :returns: a list of RDKit molecules
        """

        mols = []

        for sample in sequences:
            warheads = sample.input
            generated_linker = sample.output

            linker = attachment_points.add_attachment_point_numbers(
                generated_linker, canonicalize=False
            )
            mol: Chem.Mol = bond_maker.join_scaffolds_and_decorations(  # may return None
                linker, warheads
            )
            mols.append(mol)

        return mols


LinkinventTransformerSampler = LinkinventSampler
