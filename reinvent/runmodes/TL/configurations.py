"""Configurations for TL."""

__all__ = [
    "StepLRConfiguration",
    "LambdaLRConfiguration",
]
from dataclasses import dataclass


@dataclass(frozen=True)
class StepLRConfiguration:
    lr: float = 1e-4
    min: float = 1e-07
    gamma: float = 0.95
    step: int = 10


@dataclass(frozen=True)
class LambdaLRConfiguration:
    lr: float = 1e-4
    min: float = 1e-10
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    factor: float = 1.0
    warmup: float = 4000
