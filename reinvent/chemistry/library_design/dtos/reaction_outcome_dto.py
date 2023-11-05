from typing import List, Tuple

from dataclasses import dataclass
from rdkit.Chem.rdchem import Mol


@dataclass
class ReactionOutcomeDTO:
    reaction_smarts: str
    reaction_outcomes: List[Tuple[Mol]]
    targeted_molecule: Mol
