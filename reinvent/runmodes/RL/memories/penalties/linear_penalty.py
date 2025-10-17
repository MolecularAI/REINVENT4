from .scaffold_penalty import ScaffoldPenalty


class LinearPenalty(ScaffoldPenalty):
    """Penalize extrinsic reward using linear function."""

    def calculate_penalty(self, scaffold: str) -> float:

        bucket_size = self.scaffold_memory.max_size
        n_scaffold_instances = self.scaffold_memory.count_bucket(scaffold)

        penalty = 1 - n_scaffold_instances / bucket_size

        return max(0.0, penalty)
