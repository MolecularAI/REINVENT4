"""The LinkInvent optimization algorithm"""

from __future__ import annotations

__all__ = ["LinkinventLearning", "LinkinventTransformerLearning"]
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

        fragments = self.sampled.items2

        results = self.scoring_function(
            self.sampled.smilies, self.invalid_mask, self.duplicate_mask, fragments
        )

        return results

    def update(self, results: ScoreResults):
        if self.prior.version == 1:  # RNN-based
            return self._update_common(results)
        elif self.prior.version == 2:  # Transformer-based
            return self._update_common_transformer(results)


LinkinventTransformerLearning = LinkinventLearning
