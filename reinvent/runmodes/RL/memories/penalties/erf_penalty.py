import math

from .. import BasePenalty


class ErfPenalty(BasePenalty):
    """Penalize extrinsic reward using error function."""

    def calculate_penalty(self, bucket_utilization: int) -> float:

        return (
            1
            + math.erf(math.sqrt(math.pi) / self.bucket_size)
            - math.erf(math.sqrt(math.pi) / self.bucket_size * bucket_utilization)
        )
