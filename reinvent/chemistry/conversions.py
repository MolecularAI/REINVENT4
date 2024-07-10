"""A collection of RDKit wrappers for various types of SMILES conversions"""

import sys
import random
from typing import List, Tuple, Optional

from rdkit.Chem import (
    AllChem,
    MolFromSmiles,
    MolToSmiles,
    MolStandardize,
)
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from molvs import Standardizer


class Conversions:
    """For backward compatibility only

    Staged learning agents store the diversity filter (DF) in the agent. The DF
    in older code relies on the presence oi this class.  This class attempts to
    map the module's function into the class.
    """

    def __getattr__(self, name):
        """Resolve methods calls to module functions

        We assume here that only functions of this module will be called.
        """

        fct = getattr(sys.modules[__name__], name)
        return fct


def smiles_to_mols_and_indices(query_smiles: List[str]) -> Tuple[List[Mol], List[int]]:
    mols = [MolFromSmiles(smile) for smile in query_smiles]
    valid_mask = [mol is not None for mol in mols]
    valid_idxs = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    valid_mols = [mols[idx] for idx in valid_idxs]
    return valid_mols, valid_idxs


def mols_to_fingerprints(
    molecules: List[Mol], radius: int = 3, use_counts: bool = True, use_features: bool = True
) -> List[UIntSparseIntVect]:
    fingerprints = [
        AllChem.GetMorganFingerprint(mol, radius, useCounts=use_counts, useFeatures=use_features)
        for mol in molecules
    ]
    return fingerprints


def smiles_to_mols(query_smiles: List[str]) -> List[Mol]:
    mols = [MolFromSmiles(smile) for smile in query_smiles]
    valid_mask = [mol is not None for mol in mols]
    valid_idxs = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    valid_mols = [mols[idx] for idx in valid_idxs]
    return valid_mols


def smiles_to_fingerprints(
    query_smiles: List[str], radius=3, use_counts=True, use_features=True
) -> List[UIntSparseIntVect]:
    mols = smiles_to_mols(query_smiles)
    fingerprints = mols_to_fingerprints(
        mols, radius=radius, use_counts=use_counts, use_features=use_features
    )
    return fingerprints


def smile_to_mol(smile: str) -> Mol:
    """
    Creates a Mol object from a SMILES string.
    :param smile: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smile:
        return MolFromSmiles(smile)


def mols_to_smiles(molecules: List[Mol], isomericSmiles=False, canonical=True) -> List[str]:
    """This method assumes that all molecules are valid."""
    valid_smiles = [
        MolToSmiles(mol, isomericSmiles=isomericSmiles, canonical=canonical) for mol in molecules
    ]
    return valid_smiles


def mol_to_smiles(molecule: Mol, isomericSmiles=False, canonical=True) -> str:
    """
    Converts a Mol object into a canonical SMILES string.
    :param molecule: Mol object.
    :return: A SMILES string.
    """
    if molecule:
        return MolToSmiles(molecule, isomericSmiles=isomericSmiles, canonical=canonical)


def mol_to_random_smiles(molecule: Mol, isomericSmiles=False) -> str:
    """
    Converts a Mol object into a random SMILES string.
    :return: A SMILES string.
    """
    if molecule:
        new_atom_order = list(range(molecule.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = RenumberAtoms(molecule, newOrder=new_atom_order)
        return MolToSmiles(random_mol, canonical=False, isomericSmiles=isomericSmiles)


def convert_to_rdkit_smiles(
    smiles: str, allowTautomers=True, sanitize=False, isomericSmiles=False
) -> str:
    """
    :param smiles: Converts a smiles string into a canonical SMILES string.
    :type allowTautomers: allows having same molecule represented in different tautomeric forms
    """
    if allowTautomers:
        return MolToSmiles(MolFromSmiles(smiles, sanitize=sanitize), isomericSmiles=isomericSmiles)
    else:
        return MolStandardize.canonicalize_tautomer_smiles(smiles)


def convert_to_standardized_smiles(smiles: str) -> Optional[str]:
    """Standardize SMILES for Mol2Mol

    This should only be used to validate and transform user input
    because the code will abort execution on any error it finds.

    param smiles: single SMILES string
    return: single SMILES string
    """

    mol = MolFromSmiles(smiles, sanitize=True)

    if not mol:  # RDKit fails silently
        raise RuntimeError(f"RDKit does not accept SMILES: {smiles}")

    standardizer = Standardizer()  # MolVS

    try:
        smol = standardizer(mol)  # runs SanitizeMol() first
        smol = standardizer.charge_parent(smol)  # largest fragment uncharged
        smi = MolToSmiles(smol, isomericSmiles=True)
    except Exception as error:  # RDKit may raise multiple exceptions
        raise RuntimeError(f"RDKit does not accept SMILES: {smiles} {error}")

    # Sometimes when standardizing ChEMBL [H] are not removed so try a
    # second call
    if "[H]" in smi:
        return convert_to_standardized_smiles(smi)
    else:
        return smi


def copy_mol(molecule: Mol) -> Mol:
    """
    Copies, sanitizes, canonicalizes and cleans a molecule.
    :param molecule: A Mol object to copy.
    :return : Another Mol object copied, sanitized, canonicalized and cleaned.
    """
    return smile_to_mol(mol_to_smiles(molecule))


def randomize_smiles(smiles: str) -> str:
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param smiles: A smiles string
    :returns: A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = MolFromSmiles(smiles)
    if mol:
        new_atom_order = list(range(mol.GetNumHeavyAtoms()))
        random.shuffle(new_atom_order)
        random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
        return MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
