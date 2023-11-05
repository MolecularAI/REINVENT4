"""Define the state"""

from __future__ import annotations

__all__ = ["ModelState"]
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter
    from reinvent.runmodes.RL.memories import DiversityFilter


@dataclass
class ModelState:
    """Data structures that describe a model's state"""

    agent: ModelAdapter
    diversity_filter: DiversityFilter

    # FIXME: dataclasses.asdict does not work here (deepcopy of tensors not
    #        created explicitly by the user not support by pytorch)
    def as_dict(self):
        return dict(agent=self.agent, diversity_filter=self.diversity_filter)
