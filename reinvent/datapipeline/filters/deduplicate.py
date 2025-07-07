"""SMILES deduplication"""

__all__ = ["inchi_key_deduplicator"]
from typing import Sequence

from rdkit.Chem import MolToInchiKey, Mol


def inchi_key_deduplicator(mols: Sequence[Mol], smilies: Sequence[str]) -> list[str]:
    """Deduplicate using the InchiKey

    :param mols: RDKit molecules
    :param smilies: SMILES associated with the molecules
    :returns: deudplicate list of SMILES
    """

    data = {}

    for smiles, mol in zip(smilies, mols):
        if not smiles:
            continue

        inchi_key = MolToInchiKey(mol)
        data[inchi_key] = smiles  # keeps the last occurrence

    return list(data.values())
