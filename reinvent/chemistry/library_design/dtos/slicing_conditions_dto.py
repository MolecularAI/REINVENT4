from typing import List

from dataclasses import dataclass

from reinvent.chemistry.library_design.dtos.filtering_condition_dto import FilteringConditionDTO


@dataclass
class SlicingConditionsDTO:
    scaffold: List[FilteringConditionDTO]
    decoration: List[FilteringConditionDTO]
