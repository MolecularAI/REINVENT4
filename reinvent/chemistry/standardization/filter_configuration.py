from typing import Dict, Any

from dataclasses import dataclass


@dataclass
class FilterConfiguration:
    name: str
    parameters: Any
