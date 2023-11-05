from dataclasses import dataclass


@dataclass
class FailedReactionDTO:
    reaction_smirks: str
    molecule_smiles: str
