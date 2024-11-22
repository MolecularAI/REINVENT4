"""The Mol2Mol optimization algorithm"""

from __future__ import annotations

__all__ = ["Mol2MolLearning"]
import logging
from typing import TYPE_CHECKING

from .learning import Learning
from .distance_penalty import score as _score

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults

logger = logging.getLogger(__name__)


class Mol2MolLearning(Learning):
    """Mol2Mol optimization"""

    def score(self):
        return _score(self)

    def update(self, results: ScoreResults):
        return self._update_common_transformer(results)
