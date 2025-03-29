"""Base class for optimization to hold common functionality

This basically follows the Template Method pattern.  The (partially) abstract
base class holds the common functionality while the concrete implementation
take care of the specifics for optimization of the model.
"""

from __future__ import annotations

__all__ = ["Learning"]
import logging
import time
from typing import List, TYPE_CHECKING, Optional
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

try:
    from iSIM.comp import calculate_isim
    from iSIM.utils import binary_fps

    have_isim = True
except ImportError:
    have_isim = False

from .reports import RLTBReporter, RLCSVReporter, RLRemoteReporter, RLReportData
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.utils import get_reporter
from reinvent_plugins.normalizers.rdkit_smiles import normalize

if TYPE_CHECKING:
    from reinvent.runmodes.samplers import Sampler
    from reinvent.runmodes.RL import RLReward, terminator_callable
    from reinvent.runmodes.RL.memories import Inception
    from reinvent.models import ModelAdapter
    from reinvent.scoring import Scorer, ScoreResults

logger = logging.getLogger(__name__)


class Learning(ABC):
    """Partially abstract base class for the Template Method pattern"""

    # FIXME: too many arguments
    def __init__(
        self,
        max_steps: int,
        stage_no: int,
        prior: ModelAdapter,
        state: ModelState,
        scoring_function: Scorer,
        reward_strategy: RLReward,
        sampling_model: Sampler,
        smilies: List[str],
        distance_threshold: int,
        rdkit_smiles_flags: dict,
        inception: Inception = None,
        responder_config: dict = None,
        tb_logdir: str = None,
        tb_isim: bool = False,
    ):
        """Setup of the common framework"""

        self.max_steps = max_steps
        self.stage_no = stage_no
        self.prior = prior

        # Seed the starting state, need update in every stage
        self._state = state
        self.inception = inception

        # Need update in every stage
        self.scoring_function = scoring_function
        self.reward_nlls = reward_strategy

        self.sampling_model = sampling_model

        self.seed_smilies = smilies
        self.distance_threshold = distance_threshold

        self.rdkit_smiles_flags = rdkit_smiles_flags
        self.isomeric = False

        if "isomericSmiles" in self.rdkit_smiles_flags:
            self.isomeric = True

        self._state_info = {"name": "staged learning", "version": 1}

        # Pass on the sampling results to the specific RL class:
        # needed in scoring and updating
        self.sampled = None
        self.invalid_mask = None
        self.duplicate_mask = None

        self.logging_frequency = 1

        if responder_config:
            self.logging_frequency = max(1, responder_config.get("frequency", 1))

        self.reporters = []
        self.tb_reporter = None
        self._setup_reporters(tb_logdir)

        self.tb_isim = None

        if have_isim:
            self.tb_isim = tb_isim

        self.start_time = 0

    def optimize(self, converged: terminator_callable) -> bool:
        """Run the multistep optimization loop

        Sample from the agent, score the SMILES, update the agent parameters.
        Log some key characteristics of the current step.

        :param converged: a callable that determines convergence
        :returns: whether max_steps has been reached
        """

        step = -1
        scaffolds = None
        self.start_time = time.time()

        for step in range(self.max_steps):
            self.sampled = self.sampling_model.sample(self.seed_smilies)
            self.invalid_mask = np.where(self.sampled.states == SmilesState.INVALID, False, True)
            self.duplicate_mask = np.where(
                self.sampled.states == SmilesState.DUPLICATE, False, True
            )

            results = self.score()
            if self.prior.model_type == "Libinvent":
                results.smilies = normalize(results.smilies, keep_all=True)

            if self._state.diversity_filter:
                df_mask = np.where(self.invalid_mask, True, False)

                scaffolds = self._state.diversity_filter.update_score(
                    results.total_scores, results.smilies, df_mask
                )

            # FIXME: check for NaNs
            #        inception filter
            agent_lls, prior_lls, augmented_nll, loss = self.update(results)

            state_dict = self._state.as_dict()
            self._state_info.update(state_dict)

            nan_idx = np.isnan(results.total_scores)
            scores = results.total_scores[~nan_idx]
            mean_scores = scores.mean()

            self.report(
                step,
                mean_scores,
                scaffolds,
                score_results=results,
                agent_lls=agent_lls,
                prior_lls=prior_lls,
                augmented_nll=augmented_nll,
                loss=float(loss),
            )

            if converged(mean_scores, step):
                logger.info(f"Terminating early in {step = }")
                break

        if self.tb_reporter:  # FIXME: context manager?
            self.tb_reporter.flush()
            self.tb_reporter.close()

        if step >= self.max_steps - 1:
            return True

        return False

    __call__ = optimize

    def get_state_dict(self) -> Optional[dict]:
        """Return the state dictionary"""

        if "agent" not in self._state_info:
            return None

        model_dict = self._state_info["agent"].get_save_dict()
        state_dict = {**model_dict}

        state_dict.update(
            staged_learning=dict(
                name=self._state_info["name"],
                version=self._state_info["version"],
                diversity_filter=self._state_info["diversity_filter"],  # FIXNE: serialization
            )
        )

        return state_dict

    @property
    def state(self):
        return self._state

    def score(self):
        """Compute the score for the SMILES strings."""

        results = self.scoring_function(
            self.sampled.smilies, self.invalid_mask, self.duplicate_mask
        )

        return results

    @abstractmethod
    def update(self, score):
        """Apply the learning strategy.

        :params score: the score from self._score()
        """

    def _update_common(self, results: ScoreResults):
        """Common update for LibInvent and LinkInvent

        :param results: scoring results object
        :return: total loss
        """

        result = self._state.agent.likelihood_smiles(self.sampled)

        agent_nlls = result.likelihood
        prior_nlls = self.prior.likelihood_smiles(self.sampled).likelihood

        # NOTE: only Reinvent has inception at the moment, would need the
        #       SMILES
        return self.reward_nlls(
            agent_nlls,
            prior_nlls,
            results.total_scores,
            self.inception,
            results.smilies,
            self._state.agent,
            np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
        )

    def _update_common_transformer(self, results: ScoreResults):
        """Common update for Transformer-based models, Mol2Mol, LibInvent and LinkInvent

        :param results: scoring results object
        :return: total loss
        """
        likelihood_dto = self._state.agent.likelihood_smiles(self.sampled)

        prior_nlls = self.prior.likelihood_smiles(self.sampled).likelihood

        agent_nlls = likelihood_dto.likelihood

        return self.reward_nlls(
            agent_nlls,
            prior_nlls,
            results.total_scores,
            self.inception,
            results.smilies,
            self._state.agent,
            np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
        )

    def _setup_reporters(self, tb_logdir):
        """Setup for reporters"""

        remote_reporter = get_reporter()
        tb_reporter = None

        if tb_logdir:
            tb_reporter = SummaryWriter(log_dir=tb_logdir)
            self.tb_reporter = tb_reporter

        self.reporters.append(RLCSVReporter(None))  # we always need this ine

        # FIXME: needs a cleaner design, maybe move to caller
        for kls, args in (
            (RLTBReporter, (tb_reporter,)),
            (RLRemoteReporter, (remote_reporter, self.logging_frequency)),
        ):
            if args[0]:
                self.reporters.append(kls(*args))

    # FIXME: still needed: molecule ID
    def report(
        self,
        step: int,
        mean_score: float,
        scaffolds,
        score_results: ScoreResults,
        agent_lls: torch.tensor,
        prior_lls: torch.tensor,
        augmented_nll: torch.tensor,
        loss: float,
    ):
        """Log the results"""

        step_no = step + 1
        diversity_filter = self._state.diversity_filter

        NLL_prior = -prior_lls.cpu().detach().numpy()
        NLL_agent = -agent_lls.cpu().detach().numpy()
        NLL_augmented = augmented_nll.cpu().detach().numpy()

        prior_mean = NLL_prior.mean()
        agent_mean = NLL_agent.mean()
        augmented_mean = NLL_augmented.mean()

        mask_valid = np.where(
            (self.sampled.states == SmilesState.VALID)
            | (self.sampled.states == SmilesState.DUPLICATE),
            True,
            False,
        )
        num_valid_smiles = sum(mask_valid)
        fract_valid_smiles = num_valid_smiles / len(mask_valid)

        mask_duplicates = self.sampled.states == SmilesState.DUPLICATE
        num_duplicate_smiles = sum(np.where(mask_duplicates, True, False))
        fract_duplicate_smiles = num_duplicate_smiles / len(mask_duplicates)

        smilies = np.array(self.sampled.smilies)[mask_valid]

        isim = None

        if self.tb_isim:
            fingerprints = binary_fps(smilies, fp_type="RDKIT", n_bits=None)
            isim = calculate_isim(fingerprints, n_ary="JT")

        if self.prior.model_type == "Libinvent":
            smilies = normalize(smilies, keep_all=True)

        mask_idx = (np.argwhere(mask_valid).flatten(),)

        report_data = RLReportData(
            step=step_no,
            stage=self.stage_no,
            smilies=smilies,
            isim=isim,  # Add isim to report_data
            scaffolds=scaffolds,
            sampled=self.sampled,
            score_results=score_results,
            prior_mean_nll=prior_mean,
            agent_mean_nll=agent_mean,
            augmented_mean_nll=augmented_mean,
            prior_nll=NLL_prior,
            agent_nll=NLL_agent,
            augmented_nll=NLL_augmented,
            loss=loss,
            fraction_valid_smiles=fract_valid_smiles,
            fraction_duplicate_smiles=fract_duplicate_smiles,
            df_memory_smilies=len(diversity_filter.smiles_memory) if diversity_filter else 0,
            bucket_max_size=(
                diversity_filter.scaffold_memory.max_size if diversity_filter else None
            ),
            num_full_buckets=(
                diversity_filter.scaffold_memory.count_full() if diversity_filter else None
            ),
            num_total_buckets=(len(diversity_filter.scaffold_memory) if diversity_filter else None),
            mean_score=mean_score,
            model_type=self._state.agent.model_type,
            start_time=self.start_time,
            n_steps=self.max_steps,
            mask_idx=mask_idx,
        )

        for reporter in self.reporters:
            reporter.submit(report_data)
