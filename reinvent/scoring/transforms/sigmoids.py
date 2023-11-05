"""Sigmoid functions

FIXME: there is only a sign change for the two sigmoid functions
"""

__all__ = ["Sigmoid", "ReverseSigmoid"]
import math
from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    low: float
    high: float
    k: float


def sigmoid(x: float, low: float, high: float, k: float) -> float:
    return math.pow(10, (10 * k * (x - (low + high) * 0.5) / (low - high)))


def reverse_sigmoid(x: float, low: float, high: float, k: float) -> float:
    try:
        return 1 / (1 + 10 ** (k * (x - (high + low) * 0.5) * 10 / (high - low)))
    except:
        return 0


class Sigmoid(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.k = params.k

    def __call__(self, values) -> np.ndarray:
        transformed = [1 / (1 + sigmoid(x, self.low, self.high, self.k)) for x in values]

        return np.array(transformed, dtype=float)


class ReverseSigmoid(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.k = params.k

    def __call__(self, values) -> np.ndarray:
        transformed = [reverse_sigmoid(x, self.low, self.high, self.k) for x in values]

        return np.array(transformed, dtype=float)
