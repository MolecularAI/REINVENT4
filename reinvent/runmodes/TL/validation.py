"""Config Validation"""

from typing import Optional

from pydantic import Field

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    num_epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    input_model_file: str
    output_model_file: str  # FIXME: consider for removal
    smiles_file: str
    sample_batch_size: int = Field(100, ge=100)
    save_every_n_epochs: int = Field(1, ge=1)
    starting_epoch: int = Field(1, ge=1)
    shuffle_each_epoch: bool = True
    num_refs: int = Field(0, ge=0)
    validation_smiles_file: Optional[str] = None
    standardize_smiles: bool = True
    randomize_smiles: bool = True
    randomize_all_smiles: bool = False
    internal_diversity: bool = False
    # learning_rate_scheduler: Optional[str] = 'StepLR'
    # optimizer: Optional[str] = 'Adam'
    max_sequence_length: int = Field(128, ge=64)

    # non-transformers
    clip_gradient_norm: float = 1.0

    # transformers
    pairs: Optional[dict] = Field(
        default_factory=lambda: {
            "type": "tanimoto",
            "upper_threshold": 1.0,
            "lower_threshold": 0.7,
            "min_cardinality": 1,
            "max_cardinality": 199,
        }
    )  # FIXME: make a configuration object
    ranking_loss_penalty: bool = False
    n_cpus: int = Field(1, ge=1)


class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class TLConfig(GlobalConfig):
    parameters: SectionParameters
    scheduler: Optional[dict] = Field(default_factory=dict)
    responder: Optional[SectionResponder] = None
