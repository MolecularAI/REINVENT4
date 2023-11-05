"""The LibInvent optimization algorithm"""

from __future__ import annotations

__all__ = ["LibinventLearning"]
from typing import TYPE_CHECKING

from .learning import Learning

if TYPE_CHECKING:
    from reinvent.scoring import ScoreResults


class LibinventLearning(Learning):
    """LibInvent optimization

    FIXME: check if implementation is still correct
    """

    def update(self, results: ScoreResults):
        return self._update_common(results)
