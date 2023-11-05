from typing import List

from dataclasses import dataclass

from reinvent.chemistry.library_design.aizynth.synthetic_pathway_dto import SyntheticPathwayDTO


@dataclass
class CollectionOfPathwaysDTO:
    input: str
    pathways: List[SyntheticPathwayDTO]
