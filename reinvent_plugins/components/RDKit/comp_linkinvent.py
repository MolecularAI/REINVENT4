"""Compute various scores with RDKit"""

import itertools
from typing import List, Callable
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem as Chem, Crippen, Descriptors, Lipinski

from ..component_results import ComponentResults

logger = logging.getLogger(__name__)


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


def num_atom_stereocenters(mol: Chem.Mol) -> int:
    # Chem.CalcNumAtomStereoCenters does not work for labeled mol
    stereo_centers = 0
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            stereo_centers += 1
    return stereo_centers


def effective_length(mol: Chem.Mol) -> int:
    # if a single atom has more than one attachment point, return 0
    attachement_indices = []
    num_attachments = 0
    for atom in mol.GetAtoms():
        if atom.HasProp("Label"):
            num_attachments += atom.GetProp("Label").count("_")
            attachement_indices.append(atom.GetIdx())
    if len(attachement_indices) < num_attachments:
        return 0
    else:
        pairs = itertools.combinations(attachement_indices, 2)
        return int(min(len(Chem.GetShortestPath(mol, i, j)) - 1 for i, j in pairs))


def graph_length(mol: Chem.Mol) -> int:
    return int(np.max(Chem.GetDistanceMatrix(mol)))


def length_ratio(mol: Chem.Mol) -> float:
    max_length = graph_length(mol)
    if max_length == 0:
        return 1
    effect_length = effective_length(mol)
    return effect_length / max_length * 100


def cap_fragment(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        # Label attachment points, use neighbor atom because atom * will be replaced by H and become implicit
        i = 1

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbor_atom = atom.GetNeighbors()[0]
                # keep track of multiple attachment points in a single atom
                label = (
                    neighbor_atom.GetProp("Label") + f"_{i}"
                    if neighbor_atom.HasProp("Label")
                    else f"attachment_{i}"
                )
                neighbor_atom.SetProp(f"Label", label)
                i += 1

        # Passivated molecule
        search_patt = Chem.MolFromSmiles("*")
        sub_patt = Chem.MolFromSmiles("[H]")
        new_mol = Chem.ReplaceSubstructs(mol, search_patt, sub_patt, replaceAll=True)[0]
        clean_mol = Chem.RemoveHs(new_mol)

        return clean_mol


def compute_scores(smilies: List[str], func: Callable) -> np.array:
    """Compute scores using a RDKit function

    :param smilies: a list of SMILES
    :param func: a callable that computes the score for a RDKit molecule
    :returns: a numpy array with the scores
    """

    scores = []

    for smiles in smilies:
        try:
            mol = cap_fragment(smiles)
            score = func(mol)
        except ValueError:
            score = np.nan
            logger.warning(f"{__name__}: Invalid SMILES {smiles}")

        scores.append(score)

    return ComponentResults([np.array(scores, dtype=float)])


cls_func_map = {
    "FragmentQed": Descriptors.qed,
    "FragmentMolecularWeight": Descriptors.MolWt,
    "FragmentTPSA": Descriptors.TPSA,
    "FragmentDistanceMatrix": Chem.GetDistanceMatrix,
    "FragmentNumAtomStereoCenters": num_atom_stereocenters,
    "FragmentHBondAcceptors": Lipinski.NumHAcceptors,
    "FragmentHBondDonors": Lipinski.NumHDonors,
    "FragmentNumRotBond": Lipinski.NumRotatableBonds,
    "FragmentCsp3": Lipinski.FractionCSP3,
    "Fragmentnumsp": num_sp,
    "Fragmentnumsp2": num_sp2,
    "Fragmentnumsp3": num_sp3,
    "FragmentEffectiveLength": effective_length,
    "FragmentGraphLength": graph_length,
    "FragmentLengthRatio": length_ratio,
    "FragmentNumHeavyAtoms": Lipinski.HeavyAtomCount,
    "FragmentNumHeteroAtoms": Lipinski.NumHeteroatoms,
    "FragmentNumRings": Lipinski.RingCount,
    "FragmentNumAromaticRings": Lipinski.NumAromaticRings,
    "FragmentNumAliphaticRings": Lipinski.NumAliphaticRings,
    "FragmentSlogP": Crippen.MolLogP,
}

for cls_name, func in cls_func_map.items():

    class Temp:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, mols: List[Chem.Mol]) -> np.array:
            return compute_scores(mols, self.desc_function.__func__)

    Temp.__name__ = cls_name
    Temp.__qualname__ = cls_name

    globals()[cls_name] = Temp
    del Temp

    cls = globals()[cls_name]
    setattr(cls, "__component", True)
    setattr(cls, "desc_function", func)
