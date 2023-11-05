from typing import List

from dataclasses import dataclass


@dataclass
class SyntheticPathwayDTO:
    precursors: List[str]