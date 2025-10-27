"""The Reinvent sampling module"""

__all__ = ["ReinventSampler"]
import logging

from rdkit import Chem, DataStructs
from dppy.finite_dpps import FiniteDPP
import numpy as np

from .sampler import Sampler, remove_duplicate_sequences, validate_smiles
from .params import SAMPLE_BATCH_SIZE
from ...models.model_factory.sample_batch import SampleBatch
from reinvent.chemistry.conversions import (
    smiles_to_mols_and_indices,
    mols_to_fingerprints,
    mols_to_scaffolds_and_indices,
    mols_to_atom_pair_fingerprints,
)
from reinvent.chemistry.similarity import (
    calculate_dice_similarity_matrix,
    calculate_tanimoto_similarity_matrix,
)

logger = logging.getLogger(__name__)


class ReinventSampler(Sampler):
    """Carry out sampling with Reinvent"""

    def sample(self, dummy) -> SampleBatch:
        """Samples the Reinvent model for the given number of SMILES

        :param dummy: Reinvent does not need SMILES input
        :returns: SampleBatch
        """

        bs = self.batch_size * 10 if self.sample_strategy == "dpp" else self.batch_size

        batch_sizes = [SAMPLE_BATCH_SIZE for _ in range(bs // SAMPLE_BATCH_SIZE)]

        if remainder := bs % SAMPLE_BATCH_SIZE:
            batch_sizes += [remainder]

        sequences = []
        for batch_size in batch_sizes:
            for batch_row in self.model.sample(batch_size):
                sequences.append(batch_row)

        if self.sample_strategy == "dpp":
            sampled_smilies = [s.output for s in sequences]

            valid_mols, valid_mols_idxs = smiles_to_mols_and_indices(sampled_smilies)

            valid_scaffolds, valid_scaffolds_idxs = mols_to_scaffolds_and_indices(
                valid_mols, topological=False
            )

            valid_idxs = [valid_mols_idxs[idx] for idx in valid_scaffolds_idxs]

            valid_mols = [valid_mols[idx] for idx in valid_scaffolds_idxs]

            fps_morgan = mols_to_fingerprints(
                valid_mols, radius=3, use_counts=True, use_features=True
            )

            fps_atom_pair = mols_to_atom_pair_fingerprints(valid_scaffolds)

            assert len(fps_morgan) == len(
                fps_atom_pair
            ), f"Fingerprint length mismatch: {len(fps_morgan)} vs {len(fps_atom_pair)}"

            dice_sim = calculate_dice_similarity_matrix(fps_atom_pair)

            tanimoto_sim = calculate_tanimoto_similarity_matrix(fps_morgan)

            likelihood_kernel = dice_sim + tanimoto_sim

            # Initialize the DPP with the kernel matrix
            dpp = FiniteDPP("likelihood", **{"L": likelihood_kernel})

            # Sample indices from the DPP
            idxs_dpp = dpp.sample_exact_k_dpp(self.batch_size)

            valid_idxs_dpp = [valid_idxs[i] for i in idxs_dpp]

            sequences = [sequences[i] for i in valid_idxs_dpp]

        sampled = SampleBatch.from_list(sequences)
        sampled.items1 = None

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled, is_reinvent=True)

        mols = [
            Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
            for smiles in sampled.output
        ]

        sampled.smilies, sampled.states = validate_smiles(mols, sampled.output)

        return sampled
