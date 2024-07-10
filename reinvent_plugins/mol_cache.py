"""Simple cache for RDKit molecules

Implemented as a decorator.
"""

__all__ = ["molcache"]

from typing import List, Callable
import logging

from rdkit import Chem

logger = logging.getLogger("reinvent")
cache = {}


def molcache(func: Callable):
    """A simple decorator to tag a class"""

    def wrapper(self, smilies: List[str]):
        mols = []

        for smiles in smilies:
            if smiles in cache:
                mol = cache[smiles]
            else:
                mol = Chem.MolFromSmiles(smiles)
                cache[smiles] = mol

                if not mol:
                    logger.warning(f"{__name__}: {smiles} could not be converted")

            mols.append(mol)

        return func(self, mols)

    return wrapper
