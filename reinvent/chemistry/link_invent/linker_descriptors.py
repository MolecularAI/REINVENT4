from rdkit.Chem import Mol, GetDistanceMatrix, rdMolDescriptors, rdchem, Descriptors

from reinvent.chemistry import Conversions, TransformationTokens
from reinvent.chemistry.link_invent.bond_breaker import BondBreaker

from reinvent.chemistry.library_design.attachment_points import AttachmentPoints
from reinvent.chemistry.link_invent.attachment_point_modifier import AttachmentPointModifier


class LinkerDescriptors:
    """Molecular descriptors specific for properties of the linker"""

    def __init__(self):
        self._bond_breaker = BondBreaker()
        self._attachment_points = AttachmentPoints()
        self._conversions = Conversions()
        self._tokens = TransformationTokens()
        self._attachment_point_modifier = AttachmentPointModifier()

    def effective_length(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        ap_idx = [i[0] for i in self._bond_breaker.get_labeled_atom_dict(linker_mol).values()]
        distance_matrix = GetDistanceMatrix(linker_mol)
        effective_linker_length = distance_matrix[ap_idx[0], ap_idx[1]]
        return int(effective_linker_length)

    def max_graph_length(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        distance_matrix = GetDistanceMatrix(linker_mol)
        max_graph_length = distance_matrix.max()
        return int(max_graph_length)

    def length_ratio(self, labeled_mol: Mol) -> float:
        """
        ratio of the maximum graph length of the linker to the effective linker length
        """
        max_length = self.max_graph_length(labeled_mol)
        effective_length = self.effective_length(labeled_mol)
        return effective_length / max_length * 100

    def num_rings(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        num_rings = rdMolDescriptors.CalcNumRings(linker_mol)
        return num_rings

    def num_aromatic_rings(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(linker_mol)
        return num_aromatic_rings

    def num_aliphatic_rings(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(linker_mol)
        return num_aliphatic_rings

    def num_sp_atoms(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        num_sp_atoms = len(
            [
                atom
                for atom in linker_mol.GetAtoms()
                if atom.GetHybridization() == rdchem.HybridizationType.SP
            ]
        )
        return num_sp_atoms

    def num_sp2_atoms(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        num_sp2_atoms = len(
            [
                atom
                for atom in linker_mol.GetAtoms()
                if atom.GetHybridization() == rdchem.HybridizationType.SP2
            ]
        )
        return num_sp2_atoms

    def num_sp3_atoms(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        num_sp3_atoms = len(
            [
                atom
                for atom in linker_mol.GetAtoms()
                if atom.GetHybridization() == rdchem.HybridizationType.SP3
            ]
        )
        return num_sp3_atoms

    def num_hbd(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        num_hbd = rdMolDescriptors.CalcNumHBD(linker_mol)
        return num_hbd

    def num_hba(self, labeled_mol: Mol) -> int:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        num_hba = rdMolDescriptors.CalcNumHBA(linker_mol)
        return num_hba

    def mol_weight(self, labeled_mol: Mol) -> float:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        mol_weight = Descriptors.MolWt(linker_mol)
        return mol_weight

    def ratio_rotatable_bonds(self, labeled_mol: Mol) -> float:
        linker_mol = self._bond_breaker.get_linker_fragment(labeled_mol)
        linker_mol = self._attachment_point_modifier.cap_linker_with_hydrogen(linker_mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(linker_mol)
        total_num_bonds = linker_mol.GetNumBonds()
        ratio = num_rotatable_bonds / total_num_bonds * 100
        return ratio

    def effective_length_from_smile(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        distance_matrix = GetDistanceMatrix(linker_mol)
        (ap_idx_0,), (ap_idx_1,) = linker_mol.GetSubstructMatches(
            self._conversions.smile_to_mol(self._tokens.ATTACHMENT_POINT_TOKEN)
        )
        effective_linker_length = distance_matrix[ap_idx_0, ap_idx_1] - 2
        # subtract connection to the attachment points to be consistent with method effective_length
        return int(effective_linker_length)

    def max_graph_length_from_smile(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        (ap_idx_0,), (ap_idx_1,) = linker_mol.GetSubstructMatches(
            self._conversions.smile_to_mol(self._tokens.ATTACHMENT_POINT_TOKEN)
        )
        distance_matrix = GetDistanceMatrix(linker_mol)
        # ignore connection from attachment point to be consistent with method max_graph_length
        distance_matrix[[ap_idx_0, ap_idx_1], :] = 0
        distance_matrix[:, [ap_idx_0, ap_idx_1]] = 0
        max_graph_length = distance_matrix.max()
        return int(max_graph_length)

    def length_ratio_from_smiles(self, linker_smile: str) -> float:
        max_length = self.max_graph_length_from_smile(linker_smile)
        effective_length = self.effective_length_from_smile(linker_smile)
        return effective_length / max_length * 100

    def num_rings_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_rings = rdMolDescriptors.CalcNumRings(linker_mol)
        return num_rings

    def num_aromatic_rings_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(linker_mol)
        return num_aromatic_rings

    def num_aliphatic_rings_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(linker_mol)
        return num_aliphatic_rings

    def num_sp_atoms_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_sp_atoms = len(
            [
                atom
                for atom in linker_mol.GetAtoms()
                if atom.GetHybridization() == rdchem.HybridizationType.SP
            ]
        )
        return num_sp_atoms

    def num_sp2_atoms_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_sp2_atoms = len(
            [
                atom
                for atom in linker_mol.GetAtoms()
                if atom.GetHybridization() == rdchem.HybridizationType.SP2
            ]
        )
        return num_sp2_atoms

    def num_sp3_atoms_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_sp3_atoms = len(
            [
                atom
                for atom in linker_mol.GetAtoms()
                if atom.GetHybridization() == rdchem.HybridizationType.SP3
            ]
        )
        return num_sp3_atoms

    def num_hbd_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_hbd = rdMolDescriptors.CalcNumHBD(linker_mol)
        return num_hbd

    def num_hba_from_smiles(self, linker_smile: str) -> int:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_hba = rdMolDescriptors.CalcNumHBA(linker_mol)
        return num_hba

    def mol_weight_from_smiles(self, linker_smile: str) -> float:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        mol_weight = Descriptors.MolWt(linker_mol)
        return mol_weight

    def ratio_rotatable_bonds_from_smiles(self, linker_smile: str) -> float:
        linker_mol = self._conversions.smile_to_mol(linker_smile)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(linker_mol)
        total_num_bonds = linker_mol.GetNumBonds()
        ratio = num_rotatable_bonds / total_num_bonds * 100
        return ratio
