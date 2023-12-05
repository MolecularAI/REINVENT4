from typing import Optional, Any

from dataclasses import dataclass


@dataclass
class FilterConfiguration:
    name: str
    parameters: Optional[Any] = None
