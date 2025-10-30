"""The Mol2Mol sampling module"""

__all__ = ["Mol2MolSampler"]
from typing import List, Tuple
import logging

from rdkit import Chem
import torch.utils.data as tud

from .sampler import Sampler, validate_smiles, remove_duplicate_sequences
from . import params
from reinvent.models.transformer.core.dataset.dataset import Dataset
from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.chemistry import conversions
from reinvent.chemistry.similarity import calculate_tanimoto as _calc_tanimoto

from ...models import SampledSequencesDTO

logger = logging.getLogger(__name__)


class Mol2MolSampler(Sampler):
    """Carry out sampling with Mol2Mol"""

    def sample(self, smilies: List[str]) -> SampleBatch:
        """Samples the Mol2Mol model for the given number of SMILES

        :param smilies: list of SMILES used for sampling
        :returns: SampleBatch
        """
        # Standardize smiles in the same way as training data
        smilies = [conversions.convert_to_standardized_smiles(smile) for smile in smilies]

        smilies = (
            [self._get_randomized_smiles(smiles) for smiles in smilies]
            if self.randomize_smiles
            else smilies
        )

        # FIXME: should probably be done by caller
        #        replace hard-coded strings
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

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled, is_mol2mol=True)

        mols = [
            Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
            for smiles in sampled.output
        ]

        sampled.smilies, sampled.states = validate_smiles(
            mols, sampled.output, isomeric=self.isomeric
        )

        return sampled

    def _get_randomized_smiles(self, smiles: str):
        input_mol = conversions.smile_to_mol(smiles)
        randomized_smile = conversions.mol_to_random_smiles(input_mol, isomericSmiles=self.isomeric)

        return randomized_smile

    def calculate_tanimoto(self, reference_smiles, smiles):
        """
        Compute Tanimoto similarity between reference_smiles and smiles,
        returns the largest if multiple reference smiles provided
        """
        specific_parameters = {"radius": 2, "use_features": False}
        ref_fingerprints = conversions.smiles_to_fingerprints(
            reference_smiles,
            radius=specific_parameters["radius"],
            use_features=specific_parameters["use_features"],
        )
        valid_mols, valid_idxs = conversions.smiles_to_mols_and_indices(smiles)
        query_fps = conversions.mols_to_fingerprints(
            valid_mols,
            radius=specific_parameters["radius"],
            use_features=specific_parameters["use_features"],
        )

        scores = _calc_tanimoto(query_fps, ref_fingerprints)

        return scores

    def check_nll(
        self, input_smiles: List[str], target_smiles: List[str]
    ) -> Tuple[List[str], List[str], List[float], List[float]]:
        """
        Compute the NLL of generating each target smiles given each input reference smiles
        :param input_smiles: list of input SMILES
        :param target_smiles: list of target SMILES
        :returns: list of input SMILES, target SMILES, Tanimoto similarity, NLL
        """
        # Prepare input for checking likelihood_smiles
        dto_list = []
        for compound in input_smiles:
            for smi in target_smiles:
                current_smi = smi

                try:
                    cano_smi = conversions.convert_to_rdkit_smiles(
                        smi, sanitize=True, isomericSmiles=True
                    )
                    current_smi = cano_smi
                except Exception:
                    logger.warning(f"SMILES {smi} is invalid")

                tokenizer = SMILESTokenizer()
                try:
                    tokenized_smi = tokenizer.tokenize(current_smi)
                    self.model.vocabulary.encode(tokenized_smi)
                except KeyError as e:
                    logger.warning(
                        f"SMILES {current_smi} contains an invalid token {e}. It will be ignored"
                    )
                else:
                    dto_list.append(SampledSequencesDTO(compound, current_smi, 0))

        # Check NLL of provided target smiles given input, smiles with unknown tokens will be ignored
        i = 0
        nlls = []
        while i < len(dto_list):
            i_right = min(len(dto_list), i + params.DATALOADER_BATCHSIZE)
            if i < i_right:
                batch_dto_list = dto_list[i:i_right]
                batch_results = self.model.likelihood_smiles(batch_dto_list)
                nlls.extend(batch_results.likelihood.cpu().detach().numpy())
                i = min(len(dto_list), i + params.DATALOADER_BATCHSIZE)
            else:
                break

        input = [dto.input for dto in dto_list]
        target = [dto.output for dto in dto_list]

        # Compute Tanimoto
        valid_mols, valid_idxs = conversions.smiles_to_mols_and_indices(target)
        valid_scores = self.calculate_tanimoto(input, target)
        tanimoto = [None] * len(target)
        for i, j in enumerate(valid_idxs):
            tanimoto[j] = valid_scores[i]

        return input, target, tanimoto, nlls
