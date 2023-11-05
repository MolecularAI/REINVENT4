"""Step functions"""

__all__ = ["RightStep", "LeftStep", "Step"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    low: float = 0.0
    high: float = 0.0


class RightStep(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.high = params.high

    def __call__(self, values) -> np.ndarray:
        transformed = [1.0 if x >= self.high else 0.0 for x in values]

        return np.array(transformed, dtype=float)


class LeftStep(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low

    def __call__(self, values) -> np.ndarray:
        transformed = [1.0 if x <= self.low else 0.0 for x in values]

        return np.array(transformed, dtype=float)


class Step(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high

    def __call__(self, values) -> np.ndarray:
        transformed = [1.0 if self.low <= x <= self.high else 0.0 for x in values]

        return np.array(transformed, dtype=float)
