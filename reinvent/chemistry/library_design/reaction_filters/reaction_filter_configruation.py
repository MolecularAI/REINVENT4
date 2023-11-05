from dataclasses import dataclass
from typing import List


@dataclass
class ReactionFilterConfiguration:
    type: str
    reactions: List[List[str]]
    reaction_definition_file: str = None
