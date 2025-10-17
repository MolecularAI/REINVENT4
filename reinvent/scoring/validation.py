"""Scoring components Config"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

MAX_ALLOWED_CPU = 40  # arbitrary limit


class ScorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    filename: Optional[str] = None
    filetype: Optional[str] = None
    component: List[dict[str, dict]]
    parallel: int = Field(1, ge=1, le=MAX_ALLOWED_CPU)
    use_pumas: bool = False
