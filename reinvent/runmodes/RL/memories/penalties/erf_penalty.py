import math

from .scaffold_penalty import ScaffoldPenalty


class ErfPenalty(ScaffoldPenalty):
    """Penalize extrinsic reward using error function."""

    def calculate_penalty(self, scaffold: str) -> float:

        bucket_size = self.scaffold_memory.max_size
        n_scaffold_instances = self.scaffold_memory.count_bucket(scaffold)

        return (
            1
            + math.erf(math.sqrt(math.pi) / bucket_size)
            - math.erf(math.sqrt(math.pi) / bucket_size * n_scaffold_instances)
        )
