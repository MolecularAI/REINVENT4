"""Sigmoid functions"""

__all__ = ["DoubleSigmoid"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform
from .sigmoid_functions import double_sigmoid


@dataclass
class Parameters:
    type: str
    low: float
    high: float
    coef_div: float = 100.0
    coef_si: float = 150.0
    coef_se: float = 150.0


class DoubleSigmoid(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.coef_div = params.coef_div
        self.coef_si = params.coef_si
        self.coef_se = params.coef_se

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)
        transformed = double_sigmoid(
            values, self.low, self.high, self.coef_div, self.coef_si, self.coef_se
        )

        return transformed
