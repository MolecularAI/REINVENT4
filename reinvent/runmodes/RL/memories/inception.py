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
import time
from enum import IntEnum
import logging

import torch
import numpy as np

from .utils.inception_results import InceptionResults
from .diversity_filter import DiversityFilter

logger = logging.getLogger(__name__)


class Order(IntEnum):
    """Storage order"""

    SAMPLED_SMILES = 0
    SCORES = 1
    LLS = 2
    AGENT_LLS = 3


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
        diversity_filter: DiversityFilter | None = None,
        diversity_penalty_weight: float = 0.0,
        is_weighted_sampling: bool = True,
        is_weight_clip: float | None = None,
        is_weight_temperature: float = 1.0,
        debug: bool = False,
    ):
        """Inception setup

        :param memory_size: SMILES memory size
        :param sample_size: number of SMILES to be sampled
        :param seed_smilies: list of seed SMILES
        :param scoring_function: scoring function for inception memory ordering
        :param prior: prior model
        :param is_weighted_sampling: use IS-weighted sampling when agent is available
        :param is_weight_clip: clamp log IS weights to [-clip, clip] before softmax
        :param is_weight_temperature: temperature divisor for log IS weights
        """

        self.maxsize = memory_size
        self.sample_size = sample_size
        self.seed_smilies = seed_smilies
        self.scoring_function = scoring_function
        self.prior = prior
        self.is_weighted_sampling = is_weighted_sampling
        self.is_weight_clip = is_weight_clip
        self.is_weight_temperature = is_weight_temperature
        self.debug = debug

        self.diversity_filter = diversity_filter
        self.diversity_penalty_weight = diversity_penalty_weight
        self.storage = []  # stores maxsize of data for smiles
        self._storage_smilies = set()
        self.step = 0

    def __call__(
        self,
        orig_smilies: np.ndarray,
        scores: torch.Tensor,
        prior_lls: torch.Tensor,
        agent_lls: torch.Tensor = None,
        agent=None,
    ) -> tuple:
        """Compute the top scoring molecules.

        :param orig_smilies: the current SMILES directly sampled from the model, needed for
                             deduplication only
        :param scores: the aggregation scores from scoring, needed for ordering
        :param prior_lls: the prior's log likelihoods, stored for reward function
        :param agent_lls: the agent's log likelihoods at the current step, stored for IS
                          weighting; if None the prior log likelihoods are used as a proxy
        :param agent: agent model used to compute current NLLs for importance-sampling
                      weighted sampling; if None uniform random sampling is used
        :returns: the SMILES, scores and prior NLLs from the top scoring SMILES in the
                  inception memory
        """

        t0 = time.perf_counter()

        result = self.add(orig_smilies, scores, prior_lls, agent_lls)
        self.step += 1

        sampled, is_weight_max, is_weight_entropy, mean_agent_ll_drift = self.sample(agent=agent)

        if self.debug:
            runtime = time.perf_counter() - t0

            # Build results
            all_scores = [float(e[Order.SCORES]) for e in self.storage]

            num_new_added, num_evicted = result if result is not None else (None, None)

            results = InceptionResults(
                runtime=runtime,
                memory_size=len(self.storage),
                memory_capacity=self.maxsize,
                num_new_added=num_new_added,
                num_evicted=num_evicted,
                top_score=all_scores[0] if all_scores else None,
                mean_score=float(np.mean(all_scores)) if all_scores else None,
                bottom_score=all_scores[-1] if all_scores else None,
                top_smiles=self.storage[0][Order.SAMPLED_SMILES] if self.storage else None,
                is_weight_max=is_weight_max,
                is_weight_entropy=is_weight_entropy,
                mean_agent_ll_drift=mean_agent_ll_drift,
            )

            if sampled is not None:
                sampled_scores = [float(s) for s in sampled[1]]
                results.mean_sampled_score = float(np.mean(sampled_scores)) if sampled_scores else None

            # store aggregated results for later retrieval in reporting
            self.last_results = results

        return sampled

    def add(
        self,
        orig_smilies: np.ndarray,
        scores: torch.Tensor,
        lls: torch.Tensor,
        agent_lls: torch.Tensor = None,
    ) -> None |tuple[int, int]:
        """Add new data to the memory

        :param orig_smilies: SMILES to add to storage
        :param scores: scores to add to storage
        :param lls: prior log-likelihoods to add to storage
        :param agent_lls: agent log-likelihoods at add time; if None the prior lls are used
                          as a proxy (suitable for seed SMILES loaded before training begins)
        :returns: (num_new_added, num_evicted)
        """

        return self._to_internal_order(orig_smilies, scores, lls, agent_lls)

    def sample(self, agent=None) -> tuple:
        """Return a sample of given size from the top scorers.

        When *agent* is supplied the sample is drawn with probability proportional
        to the importance-sampling weight

            w_i = P_current(x_i) / P_stored(x_i)
                = exp(log_P_current(x_i) - log_P_stored(x_i))

        where ``log_P_stored`` is the agent log-likelihood recorded when x_i was
        first added to the buffer.  This upweights molecules whose probability
        under the current policy has increased relative to the stored snapshot,
        giving the buffer natural off-policy correction.

        When *agent* is None (or no stored agent log-likelihoods are available)
        the method falls back to uniform random sampling.

        :param agent: optional agent model; if provided IS-weighted sampling is used
        :returns: tuple of (SMILES, scores, prior_lls)
        """

        if not self.storage:
            return None, None, None, None

        sample_size = min(self.sample_size, len(self.storage))
        is_weight_max = None
        is_weight_entropy = None
        mean_agent_ll_drift = None

        if agent is not None and self.is_weighted_sampling:
            all_smilies = [e[Order.SAMPLED_SMILES] for e in self.storage]
            stored_agent_lls = [float(e[Order.AGENT_LLS]) for e in self.storage]

            with torch.no_grad():
                lls = agent.likelihood_smiles(all_smilies)
                current_agent_nlls = lls if isinstance(lls, torch.Tensor) else lls.likelihood
                current_agent_lls = -current_agent_nlls.cpu()

            stored = torch.tensor(stored_agent_lls, dtype=current_agent_lls.dtype)
            # IS weight: exp(log P_current - log P_stored)
            log_weights = (current_agent_lls - stored) / self.is_weight_temperature
            if self.is_weight_clip is not None:
                log_weights = log_weights.clamp(-self.is_weight_clip, self.is_weight_clip)
            # softmax gives a numerically stable normalised probability distribution
            weights = torch.softmax(log_weights, dim=0)

            is_weight_max = float(weights.max())
            # entropy: -sum(w * log(w)), clamped to avoid log(0)
            log_w = torch.log(weights.clamp(min=1e-12))
            is_weight_entropy = float(-(weights * log_w).sum())
            # mean absolute drift between current and stored agent LLs
            mean_agent_ll_drift = float((current_agent_lls - stored).abs().mean())

            indices = torch.multinomial(weights, sample_size, replacement=False).tolist()
            seq = [self.storage[i] for i in indices]
        else:
            seq = random.sample(self.storage, sample_size)

        sampled = self._from_internal_order(seq)

        return sampled, is_weight_max, is_weight_entropy, mean_agent_ll_drift

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

            # Use prior lls as proxy for agent lls at initialisation time
            self.add(standardized, scores, lls, lls)
            self._storage_smilies.update(standardized)  # NOTE: writing to global variable!

    def _to_internal_order(
        self,
        orig_smilies: np.ndarray,
        scores: torch.Tensor,
        lls: torch.Tensor,
        agent_lls=None,
    ) -> None | tuple[int, int]:
        """Keep internal order

        The score and likelihood are stored in transposed form and are kept
        in sorted order.  Sorting is done on the scores with highest score
        first.  *agent_lls* records the agent log-likelihood at add time and
        is used later for importance-sampling weight computation.

        :returns: (num_new_added, num_evicted)
        """

        if agent_lls is None:
            agent_lls = lls  # prior lls used as proxy (e.g. for seed SMILES)

        storage = []
        smiles_before = set(self._storage_smilies)

        if len(self.storage) > 1 and self.diversity_filter is not None:
            # Apply diversity penalty to existing storage 
            stored_scores = np.array([e[Order.SCORES] for e in self.storage])
            stored_smilies = [e[Order.SAMPLED_SMILES] for e in self.storage]

            logger.debug(f"Inception: applying diversity penalty to {len(stored_smilies)} stored SMILES")

            penalties, _ = self.diversity_filter.calculate_penalty(stored_scores, stored_smilies)

            # apply weight set in config to dampen effects of diversity penalty (0.0 = no effect, 1.0 = full effect)
            penalties = self.diversity_penalty_weight * penalties + (1.0 - self.diversity_penalty_weight)

            # Update scores in storage with penalties
            for i, e in enumerate(self.storage):
                self.storage[i] = (
                    stored_smilies[i],
                    stored_scores[i] * penalties[i],
                    e[Order.LLS],
                    e[Order.AGENT_LLS],
                )

        if self.step < 1:
            uniq, idx = np.unique(orig_smilies, return_index=True)

            if len(uniq) < len(orig_smilies):
                logger.debug(f"Inception: duplicated SMILES found in first batch")

            orig_smilies = uniq
            scores = scores[idx]
            lls = lls[idx]
            agent_lls = agent_lls[idx]

        for orig_smiles, score, ll, agent_ll in zip(orig_smilies, scores, lls, agent_lls):
            if orig_smiles not in self._storage_smilies:
                storage.append((orig_smiles, score, ll, agent_ll))

        self.storage.extend(storage)
        seq = sorted(self.storage, key=lambda row: row[Order.SCORES], reverse=True)
        self.storage = seq[: self.maxsize]
        self._storage_smilies = set([e[Order.SAMPLED_SMILES] for e in self.storage])

        if not self.debug:
            # skip calculation of debug metrics if not requested
            return 
        
        # Count buffer changes after top-k truncation.
        num_new_added = len(self._storage_smilies - smiles_before)
        num_evicted = len(smiles_before - self._storage_smilies)

        if logger.parent.level <= logging.DEBUG and self.storage:
            first = self.storage[0]
            smiles = first[Order.SAMPLED_SMILES]
            score = first[Order.SCORES]
            ll = first[Order.LLS]
            logger.debug(f"Inception top score: {smiles} {score:.5f} {ll:.2f}")

        return num_new_added, num_evicted

    def _from_internal_order(self, seq) -> tuple:
        """Return original order

        Order is: SMILES, scores, prior LLs, stored agent LLs
        """

        transpose = tuple(zip(*seq))

        return transpose

    def __len__(self) -> int:
        return len(self.storage)
