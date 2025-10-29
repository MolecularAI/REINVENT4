"""The LinkInvent sampling module"""

__all__ = ["LinkinventSampler", "LinkinventTransformerSampler"]
from typing import List
import logging

import torch.utils.data as tud
from rdkit import Chem
from dppy.finite_dpps import FiniteDPP

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.linkinvent.dataset.dataset import Dataset
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.chemistry import conversions, tokens
from reinvent.chemistry.library_design import attachment_points, bond_maker
from ...models.transformer.core.dataset.dataset import Dataset as TransformerDataset
from reinvent.chemistry.conversions import (
    mols_to_scaffolds_and_indices,
    mols_to_atom_pair_fingerprints,
)
from reinvent.chemistry.similarity import (
    calculate_dice_similarity_matrix,
)

logger = logging.getLogger(__name__)


class LinkinventSampler(Sampler):
    """Carry out sampling with LinkInvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: SampleBatch
        """

        n_input_smilies = len(smilies)

        if self.model.version == 2:  # Transformer-based
            smilies = self._standardize_input(smilies)

        warheads_list = self._get_randomized_smiles(smilies) if self.randomize_smiles else smilies

        clean_warheads = [
            attachment_points.remove_attachment_point_numbers(warheads)
            for warheads in warheads_list
            if warheads
        ]

        batch_size = self.batch_size * 10 if self.sample_strategy == "dpp" else self.batch_size

        if self.model.version == 1:  # RNN-based
            clean_warheads = clean_warheads * batch_size

            dataset = Dataset(clean_warheads, self.model.get_vocabulary().input)
        elif self.model.version == 2:  # Transformer-based
            if self.sample_strategy in ["multinomial", "dpp"]:
                clean_warheads = clean_warheads * batch_size

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
                sampled = self.model.sample(
                    inputs,
                    input_info,
                    "multinomial" if self.sample_strategy == "dpp" else self.sample_strategy,
                )

            for batch_row in sampled:
                sequences.append(batch_row)

        if self.sample_strategy == "dpp":

            sequences_dpp = []

            # Process each input smile separately to get diversity among inputs
            for i in range(n_input_smilies):

                seqs = sequences[i::n_input_smilies]

                mols = self._join_fragments(seqs)

                valid_mask = [mol is not None for mol in mols]
                valid_mols_idxs = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
                valid_mols = [mols[idx] for idx in valid_mols_idxs]

                valid_scaffolds, valid_scaffolds_idxs = mols_to_scaffolds_and_indices(
                    valid_mols, topological=False
                )

                valid_idxs = [valid_mols_idxs[idx] for idx in valid_scaffolds_idxs]

                valid_mols = [valid_mols[idx] for idx in valid_scaffolds_idxs]

                fps_atom_pair = mols_to_atom_pair_fingerprints(valid_scaffolds)

                dice_sim = calculate_dice_similarity_matrix(fps_atom_pair)

                likelihood_kernel = dice_sim

                # Initialize the DPP with the kernel matrix
                dpp = FiniteDPP("likelihood", **{"L": likelihood_kernel})

                # Sample indices from the DPP
                idxs_dpp = dpp.sample_exact_k_dpp(self.batch_size)

                valid_idxs_dpp = [valid_idxs[i] for i in idxs_dpp]

                sequences_dpp.extend([seqs[i] for i in valid_idxs_dpp])

            sequences = sequences_dpp

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
