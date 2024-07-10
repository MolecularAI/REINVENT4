"""Compute various scores with RDKit"""

from typing import List, Callable

import numpy as np
from rdkit.Chem import AllChem as Chem, Crippen, Descriptors, Lipinski

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache


def compute_scores(mols: List[Chem.Mol], func: Callable) -> np.array:
    """Compute scores using a RDKit function

    :param mols: a list of RDKit molecules
    :param func: a callable that computes the score for a RDKit molecule
    :returns: a numpy array with the scores
    """

    scores = []

    for mol in mols:
        try:
            score = func(mol)
        except ValueError:
            score = np.nan

        scores.append(score)

    return ComponentResults([np.array(scores, dtype=float)])


def num_sp(mol: Chem.Mol) -> int:
    num_sp_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP]
    )

    return num_sp_atoms


def num_sp2(mol: Chem.Mol) -> int:
    num_sp2_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2]
    )

    return num_sp2_atoms


def num_sp3(mol: Chem.Mol) -> int:
    num_sp3_atoms = len(
        [atom for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3]
    )
    return num_sp3_atoms


def graph_length(mol: Chem.Mol) -> int:
    return int(np.max(Chem.GetDistanceMatrix(mol)))


cls_func_map = {
    "Qed": Descriptors.qed,
    "MolecularWeight": Descriptors.MolWt,
    "GraphLength": graph_length,
    "NumAtomStereoCenters": Chem.CalcNumAtomStereoCenters,
    "HBondAcceptors": Lipinski.NumHAcceptors,
    "HBondDonors": Lipinski.NumHDonors,
    "NumRotBond": Lipinski.NumRotatableBonds,
    "Csp3": Lipinski.FractionCSP3,
    "numsp": num_sp,
    "numsp2": num_sp2,
    "numsp3": num_sp3,
    "NumHeavyAtoms": Lipinski.HeavyAtomCount,
    "NumHeteroAtoms": Lipinski.NumHeteroatoms,
    "NumRings": Lipinski.RingCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "SlogP": Crippen.MolLogP,
}

for cls_name, func in cls_func_map.items():

    class Temp:
        def __init__(self, *args, **kwargs):
            pass

        @molcache
        def __call__(self, mols: List[Chem.Mol]) -> np.array:
            return compute_scores(mols, self.desc_function.__func__)

    Temp.__name__ = cls_name
    Temp.__qualname__ = cls_name

    globals()[cls_name] = Temp
    del Temp

    cls = globals()[cls_name]
    setattr(cls, "__component", True)
    setattr(cls, "desc_function", func)
