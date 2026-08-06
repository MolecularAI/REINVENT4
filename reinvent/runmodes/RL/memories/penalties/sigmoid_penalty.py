import math

from ..penalty_base import BasePenalty


class SigmoidPenalty(BasePenalty):
    """Penalize extrinsic reward using sigmoid function."""

    def calculate_penalty(self, bucket_utilization: int) -> float:


        exponent = bucket_utilization / self.bucket_size
        exponent *= 2
        exponent -= 1
        exponent /= 0.15
        exponent *= -1

        sigmoid = 1 / (1 + math.exp(exponent))

        penalty = 1 - sigmoid

        return penalty
