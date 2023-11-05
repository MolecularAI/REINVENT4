"""Sigmoid functions"""

__all__ = ["DoubleSigmoid"]
from dataclasses import dataclass

import numpy as np

from .transform import Transform


@dataclass
class Parameters:
    type: str
    low: float
    high: float
    coef_div: float = 100.0
    coef_si: float = 150.0
    coef_se: float = 150.0


def double_sigmoid(
    x: float, low: float, high: float, coef_div: float, coef_si: float, coef_se: float
):
    try:
        A = 10 ** (coef_se * (x / coef_div))
        B = 10 ** (coef_se * (x / coef_div)) + 10 ** (coef_se * (low / coef_div))
        C = 10 ** (coef_si * (x / coef_div)) / (
            10 ** (coef_si * (x / coef_div)) + 10 ** (coef_si * (high / coef_div))
        )

        return (A / B) - C
    except:
        return 0.0


class DoubleSigmoid(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.low = params.low
        self.high = params.high
        self.coef_div = params.coef_div
        self.coef_si = params.coef_si
        self.coef_se = params.coef_se

    def __call__(self, values) -> np.ndarray:
        transformed = [
            double_sigmoid(val, self.low, self.high, self.coef_div, self.coef_si, self.coef_se)
            for val in values
        ]

        return np.array(transformed, dtype=np.float32)
