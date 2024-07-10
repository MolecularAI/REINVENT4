"""Config Validation"""

from pydantic import Field

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    smiles_file: str
    output_csv: str = "score_results.csv"
    smiles_column: str = "SMILES"
    standardize_smiles: bool = True


class ScoringConfig(GlobalConfig):
    parameters: SectionParameters
    scoring: dict = Field(default_factory=dict)  # validate in Scorer
