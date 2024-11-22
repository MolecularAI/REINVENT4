"""
Exponential decay, or exp(-x).

For values x < 0, the output is 1.0 ("rectified" of "clamped" exponential decay).
"""

__all__ = ["ExponentialDecay"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    k: float


def expdecay(x, k=1.0):
    return np.where(x < 0, 1, np.exp(-k * x))


class ExponentialDecay(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.k = params.k

        if self.k <= 0:
            raise ValueError(f"ExponentialDecay Transform: k must be > 0, got {self.k}")

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)
        transformed = expdecay(values, self.k)
        return transformed
