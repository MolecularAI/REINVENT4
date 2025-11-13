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

    def update(self, results: ScoreResults, orig_smilies):
        """Run the learning strategy"""

        agent_nlls = self._state.agent.likelihood_smiles(self.sampled.items2)
        prior_nlls = self.prior.likelihood_smiles(self.sampled.items2)

        return self.reward_nlls(
            orig_smilies,
            results.total_scores,
            agent_nlls,
            prior_nlls,
            np.argwhere(self.sampled.states == SmilesState.VALID).flatten(),
            self.inception,
            self._state.agent,
        )
