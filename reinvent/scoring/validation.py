"""Scoring components Config"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ScorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    filename: Optional[str] = None
    filetype: Optional[str] = None
    component: List[dict[str, dict]]
    parallel: bool = False
