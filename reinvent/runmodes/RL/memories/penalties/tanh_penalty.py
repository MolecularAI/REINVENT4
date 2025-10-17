import math

from .scaffold_penalty import ScaffoldPenalty


class TanhPenalty(ScaffoldPenalty):
    """Penalize extrinsic reward using hyperbolic tangent function."""

    def calculate_penalty(self, scaffold: str) -> float:

        bucket_size = self.scaffold_memory.max_size
        n_scaffold_instances = self.scaffold_memory.count_bucket(scaffold)

        term = n_scaffold_instances - 1

        term /= bucket_size

        term *= 3

        penalty = 1 - math.tanh(term)

        return penalty
