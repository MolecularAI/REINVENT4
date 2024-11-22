"""The Reinvent sampling module"""

__all__ = ["ReinventSampler"]
import logging

from rdkit import Chem

from .sampler import Sampler, remove_duplicate_sequences, validate_smiles
from ...models.model_factory.sample_batch import SampleBatch

logger = logging.getLogger(__name__)


class ReinventSampler(Sampler):
    """Carry out sampling with Reinvent"""

    def sample(self, dummy) -> SampleBatch:
        """Samples the Reinvent model for the given number of SMILES

        :param dummy: Reinvent does not need SMILES input
        :returns: SampleBatch
        """

        sampled = self.model.sample(self.batch_size)
        sampled.items1 = None  # expected in sampling runmode

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled, is_reinvent=True)

        mols = [
            Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
            for smiles in sampled.output
        ]

        sampled.smilies, sampled.states = validate_smiles(mols, sampled.output)
        return sampled
