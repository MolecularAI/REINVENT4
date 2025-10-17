"""The diversity filter is a memory for repeated SMILES

Depending on the concrete filter, scaffolds or SMILES that are repeatedly found
are memorized. Filtering happens on a given minimum score.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from reinvent.models.model_factory.sample_batch import SampleBatch
import torch

from reinvent.chemistry import conversions
from .bucket_counter import BucketCounter
from . import penalties


logger = logging.getLogger(__name__)


class DiversityFilter(ABC):
    """Keep track of repeated SMILES and filter by a minimum score threshold"""

    def __init__(
        self,
        bucket_size: int,
        minscore: float,
        minsimilarity: float,
        penalty_multiplier: float,
        rdkit_smiles_flags: dict,
        penalty_function: str,
        device: torch.device,
        prior_model_file_path: str,
        learning_rate: float,
    ):
        """Set up the diversity filters.

        :param bucket_size: size of each scaffold bucket
        :param minscore: minimum score
        :param minsimilarity: minimum similarity
        :param penalty_multiplier: pnealty multiplier
        :param rdkit_smiles_flags: RDKit flags for SMILES conversion
        :param rdkit_smiles_flags: RDKit flags for canonicalization
        """

        self.bucket_size = bucket_size
        self.minscore = minscore
        self.minsimilarity = minsimilarity
        self.penalty_multiplier = penalty_multiplier
        self.rdkit_smiles_flags = rdkit_smiles_flags

        self.scaffold_memory = BucketCounter(self.bucket_size)
        self.smiles_memory = set()

        self.device = device
        self.prior_model_file_path = prior_model_file_path
        self.learning_rate = learning_rate

        penalty_class = getattr(penalties, f"{penalty_function}Penalty")

        self.penalty = penalty_class(self.scaffold_memory)  # Ensure the penalty class is loaded

        logger.info(f"Using penalty function: {self.penalty.__class__.__name__}")

    @abstractmethod
    def update_score(
        self, scores: np.ndarray, smilies: List[str], mask: np.ndarray, sampled: SampleBatch
    ) -> Tuple[List | None, np.ndarray]:
        """Update the score according to the concrete fitler.

        :param scores: an array with precomputed scores
        :param smilies: list of SMILES
        :param mask: mask for valid SMILES
        :param sampled: batch of sampled SMILES
        :return: array with the updated scores and scaffolds where available
        """

    def score_scaffolds(
        self,
        scores: np.ndarray,
        smilies: List[str],
        mask: np.ndarray,
        topological: bool,
        similar: bool = False,
    ) -> Tuple[List, np.ndarray, List[int]]:
        """Score the found scaffolds

        :param smilies: list of SMILES
        :param topological: whether the scaffold should be made generic
        :param similar: whether to use similar scaffolds
        :returns: updated scores where applicable, scaffolds found, and active indices
        """
        active_idxs = []
        scaffolds = [None] * len(smilies)
        logger.debug(f"{__name__}: {len(smilies)=}")

        for i in np.nonzero(mask)[0]:
            smiles = smilies[i]
            scaffold = self._calculate_scaffold(smiles, topological)

            if similar:  # NOTE: only for ScaffoldSimilarity
                scaffold = self._find_similar_scaffold(scaffold)

            if smiles in self.smiles_memory:
                scores[i] = 0.0

            scaffolds[i] = scaffold

            if scores[i] >= self.minscore:
                self.smiles_memory.add(smiles)
                self.scaffold_memory.add(scaffold)
                active_idxs.append(i)

                penalty = self.penalty.calculate_penalty(scaffold)

                scores[i] *= penalty

        return scaffolds, np.copy(scores), active_idxs

    def _calculate_scaffold(self, smile: str, topological: bool) -> str:
        """Compute the Murcko scaffold for the given SMILES string

        :param smile: the SMILES strings to compute the scaffold from
        :param topological: whether the scaffold should be made generic
        :returns: scaffold SMILES string
        """

        mol = conversions.smile_to_mol(smile)
        scaffold_smiles = ""

        if mol:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)

                if topological:
                    scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

                # NOTE: MolToSmiles(canonical=True) by default
                # FIXME: do not rely on default
                scaffold_smiles = conversions.mol_to_smiles(scaffold, **self.rdkit_smiles_flags)
            except ValueError:
                pass

        return scaffold_smiles

    def purge_memories(self):
        """Purge the internal scaffold and SMILES memories"""

        self.scaffold_memory = BucketCounter(self.bucket_size)
        self.smiles_memory = set()
