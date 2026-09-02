from ..penalty_base import BasePenalty


class LinearPenalty(BasePenalty):
    """Penalize extrinsic reward using linear function."""

    def calculate_penalty(self, bucket_utilization: int) -> float:


        penalty = 1 - bucket_utilization / self.bucket_size

        return max(0.0, penalty)
