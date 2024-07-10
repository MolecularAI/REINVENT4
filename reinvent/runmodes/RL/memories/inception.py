"""Inception makes use of SMILES to speed-up optimization

Inception is the idea that an initial set of SMILES will guide optimization
towards wanted structures and so speed-up learning.  The SMILES will be kept in
memory depending on their score.  Only the N most high scoring SMILES will be
retained.  This typically means that the original SMILES will be quickly
replaced by newly created SMILES during optimization.
"""

from __future__ import annotations

__all__ = ["inception_filter", "Inception"]
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Callable, TYPE_CHECKING
import logging

import torch
import numpy as np

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter


logger = logging.getLogger(__name__)


@dataclass
class InceptionMemory:
    """Simple memory for inception. The data structure is specific to inception.

    The class takes in a list of SMILES, a list of scores and a list of
    likelihoods.  Internally a single list holds this data in transposed form.
    The data will be kept in sorted order (by scores) and only the top
    scorers are stored.
    """

    maxsize: int
    storage: List[Tuple] = field(default_factory=list, init=False)
    deduplicate: bool = True  # for R3 backward compatibility: False
    _smilies: list = field(default_factory=set, repr=False, init=False)  # internal SMILES memory

    def add(self, smilies: List[str], scores: np.ndarray, lls: np.ndarray):
        """Add lists to memory in sorter order"""

        _smilies = []
        _scores = []
        _lls = []

        if self.deduplicate:  # global deduplication, R3 did local
            for smiles, score, ll in zip(smilies, scores, lls):
                if smiles not in self._smilies:
                    _smilies.append(smiles)
                    _scores.append(score)
                    _lls.append(ll)

            self._smilies.update(_smilies)
        else:
            _smilies = smilies
            _scores = scores
            _lls = lls

        self._to_internal_order(_smilies, _scores, _lls)

    def sample(self, num_samples):
        """Return a number of random samples"""

        if not self.storage:
            return None

        sample_size = min(num_samples, len(self.storage))
        seq = random.sample(self.storage, sample_size)

        return tuple(self._from_internal_order(seq))

    def _to_internal_order(self, smilies, scores, lls):
        """Keep internal order

        The SMILES, score and likelihood are stored in transposed form and
        are kept in sorted order.  Sorting is done on the scores with highest
        score first.
        """

        transpose = zip(smilies, scores, lls)
        self.storage.extend(transpose)
        seq = sorted(self.storage, key=lambda row: row[1], reverse=True)

        logger.debug(f"Inception top score: {self.storage[0]}")

        self.storage = seq[: self.maxsize]

    def _from_internal_order(self, seq):
        """Return original order"""

        # TODO: see if this can be solved more elegantly
        transpose = list(zip(*seq))
        transpose[1] = np.array(transpose[1])
        transpose[2] = np.array(transpose[2])

        return transpose

    def __len__(self):
        return len(self.storage)


class Inception:
    def __init__(
        self,
        memory_size: int,
        sample_size: int,
        smilies: List[str],
        scoring_function,
        prior,
        deduplicate: bool,
    ):
        """Inception setup

        :param memory_size: memory size
        :param sample_size:
        :param smilies: list of SMILES
        :param scoring_function:
        :param prior:
        """
        self.sample_size = sample_size
        self.smilies = smilies
        self.scoring_function = scoring_function
        self.prior = prior

        valid_smiles_idx = self._validate_smiles_to_prior_vocabulary(smilies)

        if len(valid_smiles_idx) < len(smilies):
            invalid_smilies = []

            for i, smi in enumerate(smilies):
                if i not in valid_smiles_idx:
                    invalid_smilies.append(smi)

            raise RuntimeError(
                f"Found smilies incompatible with the prior: {', '.join(invalid_smilies)}"
            )

        self.memory = InceptionMemory(maxsize=memory_size, deduplicate=deduplicate)

    def _validate_smiles_to_prior_vocabulary(self, smilies: List[str]) -> list:
        """Return the list of SMILES indices compatibles with the prior vocabulary

        : param smiles: list of SMILES
        """
        valid_idx = []
        for i, smi in enumerate(smilies):
            if smi is None:
                continue
            all_tokens_in_vocabulary = True
            for token in self.prior.tokenizer.tokenize(smi):
                all_tokens_in_vocabulary = all_tokens_in_vocabulary and (
                    token in self.prior.vocabulary.tokens()
                )
            if all_tokens_in_vocabulary:
                valid_idx.append(i)
        return valid_idx

    def _load_smilies_to_memory(self):
        if len(self.smilies):
            # NOTE: we assume that the SMILES have been standardized earlier
            standardized = [smiles for smiles in self.smilies if smiles is not None]

            # FIXME: validate in caller, check for duplicates
            filter_mask = np.full(len(standardized), True, dtype=bool)

            score = self.scoring_function(self.smilies, filter_mask, filter_mask)

            # TODO: likelihood_smiles() expects different data types
            #       depending on model e.g. List[str] for Reinvent and
            #       List[SampledSequencesDTO] for Libinvent
            likelihood = self.prior.likelihood_smiles(self.smilies)
            lls = -likelihood.detach().cpu().numpy()

            self.memory.add(standardized, score.total_scores, lls)

    def update_scoring_function(self, scoring_functiom) -> None:
        """Update the scoring function

        :param scoring_functiom: the new scoring function
        """

        self.scoring_function = scoring_functiom
        self._load_smilies_to_memory()

    def add(self, smiles: List[str], scores: torch.Tensor, lls: torch.Tensor) -> None:
        """Add new data to the memory"""
        scores = scores.detach().cpu().numpy()
        lls = lls.detach().cpu().numpy()

        valid_smiles_idx = self._validate_smiles_to_prior_vocabulary(smiles)
        if len(valid_smiles_idx) < len(smiles):
            logger.warning(
                f"Found {len(smiles)-len(valid_smiles_idx):d} of {len(smiles):d} smilies incompatible with the prior for the inception filter"
            )

        if len(valid_smiles_idx) > 0:
            smiles = [smiles[i] for i in valid_smiles_idx]
            scores = scores[valid_smiles_idx]
            lls = lls[valid_smiles_idx]

            self.memory.add(smiles, scores, lls)

    # FIXME: return type
    def sample(self):
        """Return a random sample of given size from the top scorers."""

        sampled = self.memory.sample(self.sample_size)

        if sampled:
            return sampled

        return None


def inception_filter(
    agent: ModelAdapter,
    loss: torch.Tensor,
    prior_lls: torch.Tensor,
    sigma: float,
    inception: Inception,
    scores: np.ndarray,
    mask_idx: np.ndarray,
    smilies: List[str],
    RL_strategy: Callable,
) -> torch.Tensor:
    """Compute the loss from the random SAMPLE taken from the inception memory

    :param agent: agent model
    :param loss: current loss
    :param prior_lls: thr prior's log likelihoods
    :param sigma: score amplifier
    :param inception: the inception object
    :param scores: the aggregation scores from scoring
    :param mask_idx: indices of valid SMILES
    :param smilies: the current SMILES
    :param RL_strategy: reward function to call
    :returns: updated loss
    """

    result = inception.sample()

    # FIXME: assume this only happens for the first call when no seed SMILES
    #        have been provided
    #        as the scores are zero, besiaclly discard the first batch
    if not result:
        _smilies = np.array(smilies)[mask_idx]
        nsmilies = len(_smilies)
        _scores = torch.full((nsmilies,), 0.0)
        _prior_lls = torch.full((nsmilies,), 99.0)

        inception.add(_smilies, _scores, _prior_lls)

        return loss

    inception_smilies, inception_scores, inception_prior_lls = result
    total_loss = loss

    if len(inception_smilies) > 0:
        agent_lls = -agent.likelihood_smiles(inception_smilies)

        inception_loss, _ = RL_strategy(
            agent_lls,
            torch.tensor(inception_scores).to(agent_lls),
            torch.tensor(inception_prior_lls).to(agent_lls),
            sigma,
        )

        total_loss = torch.cat((loss, inception_loss), 0)

    # filter for valid SMILES
    _smilies = np.array(smilies)[mask_idx]
    _scores = scores[mask_idx]
    _prior_lls = prior_lls[mask_idx]

    inception.add(_smilies, _scores, _prior_lls)

    return total_loss
