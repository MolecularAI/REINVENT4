"""RDKit SMILES normalizer"""

__all__ = ["normalize"]

from typing import List
import logging

from rdkit import Chem

logger = logging.getLogger("reinvent")


def normalize(smilies: List[str], keep_all: bool=False) -> List:
    """Remove annotations from SMILES

    :param smilies: list of SMILES strings
    """

    cleaned_smilies = []

    for smiles in smilies:
        mol = Chem.MolFromSmiles(smiles)

        if not mol:
            if keep_all:
                cleaned_smilies.append(smiles)

            logger.warning(f"{__name__}: {smiles} could not be converted")

            continue

        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)

        cleaned_smilies.append(Chem.MolToSmiles(mol))

    return cleaned_smilies
