"""Config Validation"""

from typing import Optional
from pydantic import Field

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    smiles_file: str
    output_csv: str = "score_results.csv"
    smiles_column: str = "SMILES"
    standardize_smiles: bool = True


class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class ScoringConfig(GlobalConfig):
    parameters: SectionParameters
    scoring: dict = Field(default_factory=dict)  # validate in Scorer
    responder: Optional[SectionResponder] = None
