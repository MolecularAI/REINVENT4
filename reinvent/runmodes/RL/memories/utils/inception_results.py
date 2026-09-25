from dataclasses import dataclass


@dataclass
class InceptionResults:
    """Capturing metrics for the Inception replay memory.

    Tracks buffer state, turnover, score statistics, and IS weight diagnostics.
    """

    runtime: float | None = None
    """Wall-clock time of the inception call in seconds."""

    memory_size: int | None = None
    """Current number of SMILES stored in the buffer."""

    memory_capacity: int | None = None
    """Maximum buffer capacity (maxsize)."""

    num_new_added: int | None = None
    """How many new unique SMILES entered the buffer this step."""

    num_evicted: int | None = None
    """How many SMILES were pushed out of the buffer this step."""

    top_score: float | None = None
    """Highest score in the buffer."""

    mean_score: float | None = None
    """Mean score across all buffer entries."""

    bottom_score: float | None = None
    """Lowest score still in the buffer (admission threshold)."""

    mean_sampled_score: float | None = None
    """Mean score of the sample returned to the loss."""

    is_weight_max: float | None = None
    """Max IS weight in the sample (None when IS-weighted sampling is off)."""

    is_weight_entropy: float | None = None
    """Entropy of the IS weight distribution (None when IS is off)."""

    top_smiles: str | None = None
    """SMILES with the highest score in the buffer."""

    mean_agent_ll_drift: float | None = None
    """Mean absolute difference between current and stored agent log-likelihoods."""
