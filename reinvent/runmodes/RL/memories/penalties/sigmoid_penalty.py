import math

from .scaffold_penalty import ScaffoldPenalty


class SigmoidPenalty(ScaffoldPenalty):
    """Penalize extrinsic reward using sigmoid function."""

    def calculate_penalty(self, scaffold: str) -> float:

        bucket_size = self.scaffold_memory.max_size
        n_scaffold_instances = self.scaffold_memory.count_bucket(scaffold)

        exponent = n_scaffold_instances / bucket_size
        exponent *= 2
        exponent -= 1
        exponent /= 0.15
        exponent *= -1

        sigmoid = 1 / (1 + math.exp(exponent))

        penalty = 1 - sigmoid

        return penalty
