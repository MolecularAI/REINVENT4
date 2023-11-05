"""The Reinvent optimization algorithm"""

from __future__ import annotations

__all__ = ["ReinventLearning"]
import logging
from typing import TYPE_CHECKING

import numpy as np

from reinvent.models.model_factory.sample_batch import SmilesState
from .learning import Learning

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults

logger = logging.getLogger(__name__)


class ReinventLearning(Learning):
    """Reinvent optimization"""

    def update(self, results: ScoreResults):
        """Run the learning strategy"""

        # NOTE: this line comes originally from the RL learning strategy
        prior_nlls = self.prior.likelihood(self.sampled.sequences)  # Tensor

        return self.reward_nlls(
            self.sampled.nlls,  # agent NLL
            prior_nlls,  # prior NLL
            results.total_scores,
            self.inception,
            results.smilies,
            self._state.agent,
            np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
        )
