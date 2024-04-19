"""Base class for optimization to hold common functionality

This basically follows the Template Method pattern.  The (partially) abstract
base class holds the common functionality while the concrete implementation
take care of the specifics for optimization of the model.
"""

from __future__ import annotations

__all__ = ["Learning"]
import logging
import time
from typing import List, TYPE_CHECKING
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ..reporter import remote
from .reports.tensorboard import TBData, write_report as tb_report
from .reports.remote import RemoteData, send_report
from .reports.csv_summmary import CSVSummary, write_summary
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent.runmodes.reporter.remote import get_reporter, NoopReporter


if TYPE_CHECKING:
    from reinvent.runmodes.samplers import Sampler
    from reinvent.runmodes.RL import RLReward, terminator_callable
    from reinvent.runmodes.RL.memories import Inception
    from reinvent.runmodes.dtos import ChemistryHelpers
    from reinvent.models import ModelAdapter
    from reinvent.scoring import Scorer, ScoreResults

logger = logging.getLogger(__name__)
remote_reporter = remote.get_reporter()


class Learning(ABC):
    """Partially abstract base class for the Template Method pattern"""

    def __init__(
        self,
        max_steps: int,
        prior: ModelAdapter,
        state: ModelState,
        scoring_function: Scorer,
        reward_strategy: RLReward,
        sampling_model: Sampler,
        smilies: List[str],
        distance_threshold: int,
        rdkit_smiles_flags: dict,
        inception: Inception = None,
        chemistry: ChemistryHelpers = None,
        responder_config: dict = None,
        tb_logdir: str = None,
    ):
        """Setup of the common framework"""

        self.max_steps = max_steps
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

        self.chemistry = chemistry

        self._state_info = {"name": "staged learning", "version": 1}

        # Pass on the sampling results to the specific RL class:
        # needed in scoring and updating
        self.sampled = None
        self.invalid_mask = None
        self.duplicate_mask = None

        self.logging_frequency = 1

        if responder_config:
            self.logging_frequency = max(1, responder_config.get("frequency", 1))

        self.tb_reporter = None

        if tb_logdir:
            self.tb_reporter = SummaryWriter(log_dir=tb_logdir)

        self.reporter = get_reporter()

        self.start_time = 0

        self.__write_csv_header = True

    def optimize(self, converged: terminator_callable) -> bool:
        """Run the multistep optimization loop

        Sample from the agent, score the SNILES, update the agent parameters.
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

            # FIXME: move this to scoring
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

        if self.tb_reporter:
            self.tb_reporter.flush()
            self.tb_reporter.close()

        if step >= self.max_steps - 1:
            return True

        return False

    __call__ = optimize

    def get_state_dict(self):
        """Return the state dictionary"""

        model_dict = self._state_info["agent"].get_save_dict()
        state_dict = {**model_dict}

        state_dict.update(
            staged_learning=dict(
                name=self._state_info["name"],
                version=self._state_info["version"],
                diversity_filter=self._state_info["diversity_filter"],
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
        _input = result.batch.input  # SMILES
        _output = result.batch.output  # SMILES

        prior_nlls = self.prior.likelihood(*_input, *_output)

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
        batch = likelihood_dto.batch

        prior_nlls = self.prior.likelihood(
            batch.input, batch.input_mask, batch.output, batch.output_mask
        )

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
        NLL_augm = augmented_nll.cpu().detach().numpy()

        prior_mean = NLL_prior.mean()
        agent_mean = NLL_agent.mean()
        augm_mean = NLL_augm.mean()

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
        mask_idx = (np.argwhere(mask_valid).flatten(),)

        if self.tb_reporter:
            tb_data = TBData(
                step_no,
                score_results,
                smilies,
                prior_nll=prior_mean,
                agent_nll=agent_mean,
                augmented_nll=augm_mean,
                loss=loss,
                fraction_valid_smiles=fract_valid_smiles,
                fraction_duplicate_smiles=fract_duplicate_smiles,
                bucket_max_size=diversity_filter.scaffold_memory.max_size
                if diversity_filter
                else None,
                num_full_buckets=diversity_filter.scaffold_memory.count_full()
                if diversity_filter
                else None,
                num_total_buckets=len(diversity_filter.scaffold_memory)
                if diversity_filter
                else None,
                mean_score=mean_score,
                mask_idx=mask_idx,
            )

            tb_report(self.tb_reporter, tb_data)

        if (step == 0 or step % self.logging_frequency == 0) and not isinstance(
            self.reporter, NoopReporter
        ):
            logger.info(
                f"remote reporting at step {step} with reporter type: {self.reporter.__class__.__name__}"
            )

            report = RemoteData(
                step_no,
                score_results,
                prior_nll=prior_mean,
                agent_nll=agent_mean,
                fraction_valid_smiles=fract_valid_smiles,
                number_of_smiles=len(diversity_filter.smiles_memory) if diversity_filter else None,
                start_time=self.start_time,
                n_steps=self.max_steps,
                mean_score=mean_score,
                mask_idx=mask_idx,
            )

            send_report(report, self.reporter)

        csv_summary = CSVSummary(
            step_no, score_results, NLL_prior, NLL_agent, NLL_augm, scaffolds, self.sampled.states
        )

        header, columns = write_summary(csv_summary, write_header=self.__write_csv_header)

        if self.__write_csv_header:
            self.__write_csv_header = False

        lines = [" | " + " ".join(header)]
        NUM_ROWS = 10  # FIXME

        for i, row in enumerate(zip(*columns)):
            if i >= NUM_ROWS:
                break

            out = []

            for item in row:
                if isinstance(item, (float, int, np.floating, np.integer)):
                    num = f"{item:.2f}"
                    out.append(num)
                elif item is None:
                    out.append("--")
                else:
                    out.append(item)

            lines.append(" | " + " ".join(out))

        lines = "\n".join(lines)

        logger.info(
            f"Score: {mean_score:.2f} Agent NLL: {agent_mean:.2f} Valid: {round(100 * fract_valid_smiles):3d}% Step: {step_no}\n"
            f"{lines}"
        )
