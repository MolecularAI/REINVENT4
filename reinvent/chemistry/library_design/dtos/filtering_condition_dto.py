from dataclasses import dataclass


@dataclass
class FilteringConditionDTO:
    name: str
    min: float = None
    max: float = None
    equals: float = None
