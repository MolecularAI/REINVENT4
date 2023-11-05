from rdkit.Chem import GetDistanceMatrix
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumRotatableBonds

import numpy as np
from rdkit.Chem.MolSurf import TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAtomStereoCenters, CalcNumAromaticRings, \
    CalcNumAliphaticRings


class PhysChemDescriptors:
    """Molecular descriptors.
    The descriptors in this class are mostly calculated RDKit phys-chem properties.
    """

    def maximum_graph_length(self, mol) -> int:
        return int(np.max(GetDistanceMatrix(mol)))

    def hba_libinski(self, mol) -> int:
        return NumHAcceptors(mol)

    def hbd_libinski(self, mol) -> int:
        return NumHDonors(mol)

    def mol_weight(self, mol) -> float:
        return MolWt(mol)

    def number_of_rings(self, mol) -> int:
        return CalcNumRings(mol)

    def number_of_aromatic_rings(self, mol) -> int:
        return CalcNumAromaticRings(mol)

    def number_of_aliphatic_rings(self, mol) -> int:
        return CalcNumAliphaticRings(mol)

    def number_of_rotatable_bonds(self, mol) -> int:
        return NumRotatableBonds(mol)

    def slog_p(self, mol) -> float:
        return MolLogP(mol)

    def tpsa(self, mol) -> float:
        return TPSA(mol)

    def number_of_stereo_centers(self, mol) -> int:
        return CalcNumAtomStereoCenters(mol)

    def number_atoms_in_largest_ring(self, mol) -> int:
        ring_info = mol.GetRingInfo()
        ring_size = [len(ring) for ring in ring_info.AtomRings()]
        max_ring_size = max(ring_size) if ring_size else 0
        return int(max_ring_size)
