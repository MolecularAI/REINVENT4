from dataclasses import dataclass
from rdkit.Chem.rdChemReactions import ChemicalReaction


@dataclass
class ReactionDTO:
    reaction_smarts: str
    chemical_reaction: ChemicalReaction
