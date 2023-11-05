import re

from rdkit.Chem import Mol
from reinvent.chemistry import Conversions


class AttachmentPointModifier:
    """Manipulate linker SMILES attachment point tokens for compatibility with rdkit calculations"""

    def __init__(self):
        self._conversions = Conversions()

    def extract_attachment_atoms(self, linker_smiles: str) -> list:
        """return a list of all attachment point atoms"""
        # extract all atoms in square brackets
        bracket_atoms = re.findall(r"\[(.*?)\]", linker_smiles)
        # extract only the linker attachment atoms - these atoms have ":" separating the atoms and the label
        attachment_atoms = [
            attachment_atom for attachment_atom in bracket_atoms if ":" in attachment_atom
        ]

        return attachment_atoms

    def add_explicit_H_to_atom(self, tokens: str) -> str:
        """modifies a SMILES sequence by incrementing the number of explicit hydrogens by 1"""
        # extract the attachment atom tokens without the position label
        attachment_tokens = tokens.split(":")[0]

        # if the attachment atom is charged, return the tokens unchanged
        if attachment_tokens[-1] == "+" or attachment_tokens[-1] == "-":
            return attachment_tokens
        # if the attachment atom contains only 1 explicit hydrogen, a number will not
        # be present in the SMILES, e.g. "CH". In this case, manually add a "2"
        elif attachment_tokens[-1] == "H":
            return attachment_tokens + str(2)
        # if the attachment atom has no explicit hydrogens and does not contain a charge, add a hydrogen
        elif "H" not in attachment_tokens:
            return attachment_tokens + "H"
        # otherwise, the attachment atom has more than 1 explicit hydrogen. Increment explicit hydrogens by 1
        else:
            return attachment_tokens[:-1] + str(int(attachment_tokens[-1]) + 1)

    def cap_linker_with_hydrogen(self, linker_mol: Mol) -> Mol:
        """cap linker attachment point atoms with an explicit hydrogen"""
        linker_smiles = self._conversions.mol_to_smiles(linker_mol)
        attachment_atoms = self.extract_attachment_atoms(linker_smiles)

        for tokens in attachment_atoms:
            modified_attachment = self.add_explicit_H_to_atom(tokens)
            linker_smiles = linker_smiles.replace(tokens, modified_attachment)

        capped_linker_mol = self._conversions.smile_to_mol(linker_smiles)

        return capped_linker_mol
