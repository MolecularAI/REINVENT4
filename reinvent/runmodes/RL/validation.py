"""Config Validation"""

from typing import List, Optional

from pydantic import Field

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    prior_file: str
    agent_file: str
    summary_csv_prefix: str = "summary"
    use_checkpoint: bool = False
    purge_memories: bool = True
    smiles_file: Optional[str] = None  # not Reinvent
    sample_strategy: Optional[str] = "multinomial"  # Transformer
    distance_threshold: int = 99999  # Transformer
    batch_size: int = 100
    randomize_smiles: bool = True
    isomeric_smiles: bool = False
    temperature: float = 1.0
    tb_isim: Optional[bool] = False  # Add iSIM tracking as optional parameter


class SectionLearningStrategy(GlobalConfig):
    type: str = "dap"
    sigma: int = 128
    rate: float = 0.0001


class SectionDiversityFilter(GlobalConfig):
    type: str
    bucket_size: int = Field(25, ge=1)  # not needed for PenalizeSameSmiles
    minscore: float = Field(0.4, ge=0.0, le=1.0)  # not needed for PenalizeSameSmiles
    minsimilarity: Optional[float] = Field(0.4, ge=0.0, le=1.0)  # ScaffoldSimilarity only
    penalty_multiplier: Optional[float] = Field(0.5, ge=0.0, le=1.0)  # PenalizeSameSmiles only
    penalty_function: str = Field("Step") # Penality function for downscoring, default: 0 or 1 based on bucket_size
    intrinsic_reward: Optional[str] = Field(None)  # Add intrinsic reward as optional parameter

    # BitBirch clustering parameters
    discard: bool = Field(False) # BitBirchDiversityFilter only
    merge_threshold: float = Field(0.65, ge=0.0, le=1.0) # BitBirchDiversityFilter only
    branching_factor: int = Field(2500, ge=1) # BitBirchDiversityFilter only
    recluster_tolerance: Optional[float] = Field(None, ge=0.0) # BitBirchDiversityFilter only
    recluster_interval: Optional[int] = Field(None, ge=1) # BitBirchDiversityFilter only

    debug: bool = False  # Includes metrics in the tensorboard logs, but slows down training. Only for debugging purposes.

    learning_rate: float = Field(
        1e-4,
        gt=0.0,
    )  # IdenticalScaffoldPenaltyRND only, defaults to 1e-4

class SectionIntrinsicPenalty(GlobalConfig):
    type: str
    penalty_function: str
    bucket_size: int = Field(25, ge=1)
    minscore: float = Field(0.4, ge=0.0, le=1.0)
    learning_rate: float = Field(
        1e-4,
        gt=0.0,
    )  # IdenticalScaffoldPenaltyRND only, defaults to 1e-4


class SectionInception(GlobalConfig):
    smiles_file: Optional[str] = None
    memory_size: int = 50
    sample_size: int = 10
    deduplicate: bool = True  # obsolete, True by default now
    diversity_penalty_weight: float = Field(0.0, ge=0.0, le=1.0)
    is_weighted_sampling: bool = False
    is_weighted_scores: bool = False
    is_weight_clip: Optional[float] = None
    is_weight_temperature: float = 1.0
    debug: bool = False  # Includes metrics in the tensorboard logs, but slows down training. Only for debugging purposes.


class SectionStage(GlobalConfig):
    max_steps: int = Field(ge=1)
    max_score: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    chkpt_file: Optional[str] = None
    termination: str = "simple"
    min_steps: Optional[int] = Field(50, ge=0)
    scoring: dict = Field(default_factory=dict)  # validate in Scorer
    diversity_filter: Optional[SectionDiversityFilter] = None


# FIXME: may only need this once
class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class RLConfig(GlobalConfig):
    parameters: SectionParameters
    stage: List[SectionStage]
    learning_strategy: Optional[SectionLearningStrategy] = None
    diversity_filter: Optional[SectionDiversityFilter] = None
    intrinsic_penalty: Optional[SectionIntrinsicPenalty] = None
    inception: Optional[SectionInception] = None
    responder: Optional[SectionResponder] = None
