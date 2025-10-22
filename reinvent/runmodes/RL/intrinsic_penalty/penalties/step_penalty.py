from .scaffold_penalty import ScaffoldPenalty


class StepPenalty(ScaffoldPenalty):
    """Penalty based on step function: return 0 if bucket is full, else return 1. Default in REINVENT."""

    def calculate_penalty(self, scaffold: str) -> float:
        if self.scaffold_memory.bucket_full(scaffold):
            return 0.0

        else:
            return 1.0
