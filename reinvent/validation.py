"""Global Config Validation"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ReinventConfig(GlobalConfig):
    run_type: str
    device: str = "cpu"
    use_cuda: Optional[bool] = Field(True, deprecated="use 'device' instead")
    tb_logdir: Optional[str] = None
    json_out_config: Optional[str] = None
    seed: Optional[int] = None
    parameters: dict

    # run mode dependent
    scoring: Optional[dict] = None  # RL, scoring
    scheduler: Optional[dict] = None  # TL
    responder: Optional[dict] = None  # Rl, TL, sampling

    # RL
    stage: Optional[list] = None
    learning_strategy: Optional[dict] = None
    diversity_filter: Optional[dict] = None
    inception: Optional[dict] = None
