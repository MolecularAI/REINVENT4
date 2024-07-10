"""Defines a work package for staged learning

Stores parameters that change for each stage,
"""

from __future__ import annotations

__all__ = ["WorkPackage"]
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from reinvent.runmodes.RL.reward import RLReward
    from reinvent.runmodes.RL.memories.diversity_filter import DiversityFilter
    from reinvent.scoring import Scorer


@dataclass(frozen=True)
class WorkPackage:
    """The inputs to each stage in a staged optimization scheme."""

    scoring_function: Scorer
    learning_strategy: RLReward
    max_steps: int
    terminator: Callable
    diversity_filter: DiversityFilter = None
    out_state_filename: str = None
