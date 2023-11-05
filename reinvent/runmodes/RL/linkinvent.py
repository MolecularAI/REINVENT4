"""The LinkInvent optimization algorithm"""

from __future__ import annotations

__all__ = ["LinkinventLearning"]
from typing import TYPE_CHECKING

import numpy as np

from .learning import Learning
from reinvent.models.model_factory.sample_batch import SmilesState

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


class LinkinventLearning(Learning):
    """LinkInvent optimization"""

    def score(self):
        """Compute the score for the SMILES strings.

        Overwrites generic method to pass on fragments.
        """

        mask = np.where(self.sampled.states == SmilesState.VALID, True, False)
        fragments = self.sampled.items2

        results = self.scoring_function(self.sampled.smilies, mask, fragments)

        return results

    def update(self, results: ScoreResults):
        return self._update_common(results)
