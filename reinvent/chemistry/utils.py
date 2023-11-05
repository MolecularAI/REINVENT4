from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import numpy as np


def get_indices_of_unique_smiles(smiles: [str]) -> np.array:
    """Returns an np.array of indices corresponding to the first entries in a list of smiles strings"""
    _, idxs = np.unique(smiles, return_index=True)
    sorted_indices = np.sort(idxs)
    return sorted_indices


def compute_scaffold(mol: Chem.Mol, *, generic: bool = True, isomeric=False) -> str:
    """Computes the scaffold for an input molecule.

    :param mol: the input molecule
    :type mol: Chem.Mol
    :param generic: whether compute the generic scaffold or not (Default: True)
    :type generic: bool
    :param isomeric: whether compute the isomeric smiles or not (Default: False)
    :type generic: bool
    :rtype: str
    """
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold, isomericSmiles=isomeric, canonical=True)


def compute_num_heavy_atoms(mol: Chem.Mol) -> int:
    """Computes the number of heavy atoms.

    :param mol: input molecule
    :type mol: Chem.Mol
    :rtype: int
    """
    if mol is not None:
        return mol.GetNumHeavyAtoms()
    return 0
