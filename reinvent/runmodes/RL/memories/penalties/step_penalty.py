from ..penalty_base import BasePenalty


class StepPenalty(BasePenalty):
    """Penalty based on step function: return 0 if bucket is full, else return 1. Default in REINVENT."""

    def calculate_penalty(self, bucket_utilization: int) -> float:
        if bucket_utilization >= self.bucket_size:
            return 0.0
        else:
            return 1.0
