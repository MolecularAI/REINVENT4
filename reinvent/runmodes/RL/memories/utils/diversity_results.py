
from dataclasses import dataclass


@dataclass
class DiversityResults:
    """Capturing metrics for DiversityFilter modules.

    capturing runtime, memory size, bucket saturation, and optionally the scaffolds.
    """

    runtime: float | None = None
    """Total wall-clock time of diversity filter in seconds."""

    memory_size: int | None = None # before df_memory_smilies
    """Size of diversity memory, how many smiles are stored."""

    bucket_max_size: int | None = None
    """Maximum capacity per diversity bucket."""

    num_full_buckets: int | None = None
    """Count of buckets that reached bucket_max_size."""

    num_active: int | None = None
    """Number of active molecules that pass score threashold"""

    num_downscored: int | None = None
    """Number of molecules that have been downscored"""

    num_total_buckets: int | None = None
    """Total buckets considered (including empty/partial)."""

    scaffolds: list[str | None] | None = None
    """Optional scaffold smiles; None if not collected."""

    intrinsic_reward: float | None = None 
    """Added to score by the intrinsic reward system"""

    mean_extrinsic_score: float | None = None 
    """Mean score before any diversity filter is applied""" 

    modified_bucket_occupation: list[tuple[list[int], int]] | None = None
    """List of tuples containing indices of molecules in each bucket and their corresponding utilization after processing the batch."""