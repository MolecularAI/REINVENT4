from typing import List

from dataclasses import dataclass

from reinvent.chemistry.library_design.reaction_definitions.building_block_pair_dto import (
    BuildingBlockPairDTO,
)


@dataclass
class BuildingBlocksForCompoundDTO:
    compound: str
    reaction_name: str
    attachment_position: int
    building_block_pairs: List[BuildingBlockPairDTO]
