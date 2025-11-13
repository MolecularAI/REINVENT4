"""Inception makes use of SMILES to speed-up optimization

Inception is the idea that an initial set of SMILES will guide optimization
towards wanted structures and so speed-up learning.  The SMILES will be kept in
memory depending on their score.  Only the N most high scoring SMILES will be
retained.  This typically means that the original SMILES will be quickly
replaced by newly created SMILES during optimization.
"""

from __future__ import annotations

__all__ = ["Inception"]
import random
from enum import IntEnum
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class Order(IntEnum):
    """Storage order"""

    SAMPLED_SMILES = 0
    SCORES = 1
    LLS = 2


class Inception:
    """Implementation of a replay memory.

    The class takes in a list of SMILES, a list of scores and a list of
    likelihoods.  Internally a single list holds this data in transposed form.
    The data will be kept in sorted order (by scores) and only the top
    scorers are stored and returned an so acts as a filter.
    """

    def __init__(
        self,
        memory_size: int,
        sample_size: int,
        seed_smilies: list[str],
        scoring_function,
        prior,
    ):
        """Inception setup

        :param memory_size: SMILES memory size
        :param sample_size: number of SMILES to be sampled
        :param seed_smilies: list of seed SMILES
        :param scoring_function: scoring function for inception memory ordering
        :param prior: prior model
        """

        self.maxsize = memory_size
        self.sample_size = sample_size
        self.seed_smilies = seed_smilies
        self.scoring_function = scoring_function
        self.prior = prior

        self.storage = []  # stores maxsize of data for smiles
        self._storage_smilies = set()
        self.step = 0

    def __call__(
        self,
        orig_smilies: np.ndarray,
        scores: torch.Tensor,
        prior_lls: torch.Tensor,
    ) -> tuple:
        """Compute the top scoring molecules.

        :param orig_smilies: the current SMILES directly sampled from the model, needed for
                             deduplication only
        :param scores: the aggregation scores from scoring, needed for ordering
        :param prior_lls: thr prior's log likelihoods, stored for reward function
        :returns: the SMILES, scores and prior NLLs form the top scoring SMILES in the
                  inception memory
        """

        self.add(orig_smilies, scores, prior_lls)
        self.step += 1

        return self.sample()

    def add(self, orig_smilies: np.ndarray, scores: torch.Tensor, lls: torch.Tensor) -> None:
        """Add new data to the memory

        :param orig_smilies: SMILES to add to storage
        :param scores: scores to add to storage
        :param lls: likelihoods to add to storage
        """

        self._to_internal_order(orig_smilies, scores, lls)

    def sample(self) -> tuple | None:
        """Return a random sample of given size from the top scorers."""

        if not self.storage:
            return None

        sample_size = min(self.sample_size, len(self.storage))
        seq = random.sample(self.storage, sample_size)
        sampled = self._from_internal_order(seq)

        return sampled

    def update(self, scoring_functiom) -> None:
        """Update the scoring function

        Supports setup with multiple scoring functions.  Also reads in
        the seed SMILES.
        NOTE: must run before first use of the inception memory.

        :param scoring_functiom: the new scoring function
        """

        self.scoring_function = scoring_functiom
        self._load_seed_smilies_to_memory()

    def _load_seed_smilies_to_memory(self) -> None:
        if len(self.seed_smilies):
            # NOTE: we assume that the SMILES have been standardized earlier
            standardized = np.array([smiles for smiles in self.seed_smilies if smiles is not None])
            filter_mask = np.full(len(standardized), True, dtype=bool)

            result = self.scoring_function(standardized, filter_mask, filter_mask)
            scores = result.total_scores

            # TODO: likelihood_smiles() expects different data types
            #       depending on model e.g. List[str] for Reinvent and
            #       List[SampledSequencesDTO] for Libinvent
            likelihood = self.prior.likelihood_smiles(self.seed_smilies)
            lls = -likelihood.cpu().numpy()

            self.add(standardized, scores, lls)
            self._storage_smilies.update(standardized)  # NOTE: writing to global variable!

    def _to_internal_order(
        self, orig_smilies: np.ndarray, scores: torch.Tensor, lls: torch.Tensor
    ) -> None:
        """Keep internal order

        The score and likelihood are stored in transposed form and are kept
        in sorted order.  Sorting is done on the scores with highest score
        first.
        """

        storage = []

        if self.step < 1:
            uniq, idx = np.unique(orig_smilies, return_index=True)

            if len(uniq) < len(orig_smilies):
                logger.debug(f"Inception: duplicated SMILES found in first batch")

            orig_smilies = uniq
            scores = scores[idx]
            lls = lls[idx]

        for orig_smiles, score, ll in zip(orig_smilies, scores, lls):
            if orig_smiles not in self._storage_smilies:
                storage.append((orig_smiles, score, ll))

        self.storage.extend(storage)
        seq = sorted(self.storage, key=lambda row: row[Order.SCORES], reverse=True)
        self.storage = seq[: self.maxsize]
        self._storage_smilies = set([e[Order.SAMPLED_SMILES] for e in self.storage])

        if logger.parent.level <= logging.DEBUG and self.storage:
            first = self.storage[0]
            smiles = first[Order.SAMPLED_SMILES]
            score = first[Order.SCORES]
            ll = first[Order.LLS]
            logger.debug(f"Inception top score: {smiles} {score:.5f} {ll:.2f}")

    def _from_internal_order(self, seq) -> tuple:
        """Return original order

        Order is: SMILES, originally sampled SMILES, scores, LLs
        """

        transpose = tuple(zip(*seq))

        return transpose

    def __len__(self) -> int:
        return len(self.storage)
