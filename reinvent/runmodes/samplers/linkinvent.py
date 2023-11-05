"""The LinkInvent sampling module"""

__all__ = ["LinkinventSampler"]
from typing import List

import torch.utils.data as tud

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.linkinvent.dataset.dataset import Dataset
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.runmodes.utils.helpers import join_fragments


class LinkinventSampler(Sampler):
    """Carry out sampling with LinkInvent"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: list of SampledSequencesDTO
        """

        warheads_list = self._get_randomized_smiles(smilies) if self.randomize_smiles else smilies
        clean_warheads = [
            self.chemistry.attachment_points.remove_attachment_point_numbers(warheads)
            for warheads in warheads_list
        ]

        clean_warheads = clean_warheads * self.batch_size
        dataset = Dataset(clean_warheads, self.model.get_vocabulary().input)

        dataloader = tud.DataLoader(
            dataset,
            batch_size=params.DATALOADER_BATCHSIZE,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        sequences = []

        for batch in dataloader:
            inputs, input_seq_lengths = batch
            sampled = self.model.sample(inputs, input_seq_lengths)

            for batch_row in sampled:
                sequences.append(batch_row)

        sampled = SampleBatch.from_list(sequences)

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled)

        mols = join_fragments(sampled, reverse=True)

        sampled.smilies, sampled.states = validate_smiles(mols)

        return sampled

    def _get_randomized_smiles(self, warhead_pair_list: List[str]):
        """Y"""

        randomized_warhead_pair_list = []

        for warhead_pair in warhead_pair_list:
            warhead_list = warhead_pair.split(self.tokens.ATTACHMENT_SEPARATOR_TOKEN)
            warhead_mol_list = [
                self.chemistry.conversions.smile_to_mol(warhead) for warhead in warhead_list
            ]
            warhead_randomized_list = [
                self.chemistry.conversions.mol_to_random_smiles(mol) for mol in warhead_mol_list
            ]
            # Note do not use self.self._bond_maker.randomize_scaffold, as it would add unwanted brackets to the
            # attachment points (which are not part of the warhead vocabulary)
            warhead_pair_randomized = self.tokens.ATTACHMENT_SEPARATOR_TOKEN.join(
                warhead_randomized_list
            )
            randomized_warhead_pair_list.append(warhead_pair_randomized)

        return randomized_warhead_pair_list
