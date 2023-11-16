"""Convert to kekulized SMILES for the Lilly tools"""

__all__ = ["normalize"]

from typing import List
import logging

from rdkit import Chem

logger = logging.getLogger("reinvent")


def normalize(smilies: List[str]) -> List:
    """Remove annotations from SMILES

    :param smilies: list of SMILES strings
    """

    cleaned_smilies = []

    for smiles in smilies:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)

        if not mol:
            logger.warning(f"{__name__}: {smiles} could not be converted")
            continue

        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)

        cleaned_smilies.append(Chem.MolToSmiles(mol, kekuleSmiles=True))

    return cleaned_smilies
