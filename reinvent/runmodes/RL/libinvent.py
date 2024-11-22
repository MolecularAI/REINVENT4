"""The LibInvent optimization algorithm"""

from __future__ import annotations

__all__ = ["LibinventLearning", "LibinventTransformerLearning"]
from typing import TYPE_CHECKING

import numpy as np

from .learning import Learning
from reinvent.models.model_factory.sample_batch import SmilesState

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


class LibinventLearning(Learning):
    """LibInvent optimization"""

    def update(self, results: ScoreResults):
        if self.prior.version == 1:  # RNN-based
            return self._update_common(results)
        elif self.prior.version == 2:  # Transformer-based
            return self._update_common_transformer(results)


LibinventTransformerLearning = LibinventLearning
