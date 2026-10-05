import math

from ..penalty_base import BasePenalty


class TanhPenalty(BasePenalty):
    """Penalize extrinsic reward using hyperbolic tangent function."""

    def calculate_penalty(self, bucket_utilization: int) -> float:

        term = bucket_utilization - 1

        term /= self.bucket_size

        term *= 3

        penalty = 1 - math.tanh(term)

        return penalty
