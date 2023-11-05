from collections import defaultdict
from typing import List

from rdkit import Chem
from rdkit.Chem import Mol, EditableMol, GetMolFrags


class BondBreaker:
    """
    breaks and identify bonds in labeled molecules / smiles
    """
    def __init__(self):
        self._mol_atom_map_number = 'molAtomMapNumber'

    def labeled_mol_into_fragment_mols(self, labeled_mol: Mol) -> List[Mol]:
        e_mol = EditableMol(labeled_mol)
        for atom_pair in self.get_bond_atoms_idx_pairs(labeled_mol):
            e_mol.RemoveBond(*atom_pair)
        mol_fragments = GetMolFrags(e_mol.GetMol(), asMols=True, sanitizeFrags=False)
        return mol_fragments

    def get_linker_fragment(self, labeled_mol: Mol):
        """
        Returns the mol of the linker (labeled), where the linker is the only fragment with two attachment points
        returns None if no linker is found
        """
        fragment_mol_list = self.labeled_mol_into_fragment_mols(labeled_mol)
        linker = None
        for fragment in fragment_mol_list:
            labeled_atom_dict = self.get_labeled_atom_dict(fragment)
            if len(labeled_atom_dict) == 2:
                linker = fragment
        return linker

    def get_bond_atoms_idx_pairs(self, labeled_mol: Mol):
        labeled_atom_dict = self.get_labeled_atom_dict(labeled_mol)
        bond_atoms_idx_list = [value for value in dict(labeled_atom_dict).values()]
        return bond_atoms_idx_list

    def get_labeled_atom_dict(self, labeled_mol: Mol):
        bonds = defaultdict(list)
        for atom in labeled_mol.GetAtoms():
            if atom.HasProp(self._mol_atom_map_number):
                bonds[atom.GetProp(self._mol_atom_map_number)].append(atom.GetIdx())
        bond_dict = dict(sorted(bonds.items()))
        return bond_dict
