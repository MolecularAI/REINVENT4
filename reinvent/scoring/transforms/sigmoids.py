"""Sigmoid functions

FIXME: there is only a sign change for the two sigmoid functions
"""

__all__ = ["Sigmoid", "ReverseSigmoid"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform
from .sigmoid_functions import hard_sigmoid, stable_sigmoid


@dataclass
class Parameters:
    type: str
    low: float
    high: float
    k: float


class Sigmoid(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.k = params.k

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)

        x = values - (self.high + self.low) / 2

        if (self.high - self.low) == 0:
            k = 10.0 * self.k
            transformed = hard_sigmoid(x, k)
        else:
            k = 10.0 * self.k / (self.high - self.low)
            transformed = stable_sigmoid(x, k)

        return transformed


class ReverseSigmoid(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.k = params.k

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)

        x = values - (self.high + self.low) / 2

        if (self.high - self.low) == 0:
            k = 10.0 * self.k
            transformed = hard_sigmoid(x, k)
        else:
            k = 10.0 * self.k / (self.high - self.low)
            transformed = stable_sigmoid(x, k)

        return 1.0 - transformed
