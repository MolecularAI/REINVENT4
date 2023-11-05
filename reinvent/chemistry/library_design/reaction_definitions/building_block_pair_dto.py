from dataclasses import dataclass


@dataclass
class BuildingBlockPairDTO:
    scaffold_block: str
    decoration_block: str
