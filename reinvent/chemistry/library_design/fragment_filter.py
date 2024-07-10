from typing import List

from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Lipinski import (
    RingCount,
    NumRotatableBonds,
    NumHAcceptors,
    NumHDonors,
    HeavyAtomCount,
)
from rdkit.Chem.rdchem import Mol

from reinvent.chemistry import tokens
from reinvent.chemistry.library_design.dtos import FilteringConditionDTO
from reinvent.chemistry.library_design.molecular_descriptors_enum import MolecularDescriptorsEnum


class FragmentFilter:
    def __init__(self, conditions: List[FilteringConditionDTO]):
        """
        Initializes a fragment filter given the conditions.
        :param conditions: Conditions to use. When None is given, everything is valid.
        """
        self._descriptors_enum = MolecularDescriptorsEnum()
        self.conditions = conditions

        self._CONDITIONS_FUNC = {
            self._descriptors_enum.HEAVY_ATOM_COUNT: HeavyAtomCount,  # pylint: disable=no-member
            self._descriptors_enum.MOLECULAR_WEIGHT: MolWt,
            self._descriptors_enum.CLOGP: MolLogP,  # pylint: disable=no-member
            self._descriptors_enum.HYDROGEN_BOND_DONORS: NumHDonors,  # pylint: disable=no-member
            self._descriptors_enum.HYDROGEN_BOND_ACCEPTORS: NumHAcceptors,  # pylint: disable=no-member
            self._descriptors_enum.ROTATABLE_BONDS: NumRotatableBonds,  # pylint: disable=no-member
            self._descriptors_enum.RING_COUNT: RingCount,  # pylint: disable=no-member
        }

    def filter(self, mol: Mol) -> bool:
        """
        Validates whether a query molecule meets all filtering criteria.
        :param mol: A molecule as a Mol object.
        :return: A boolean whether the molecule is valid or not.
        """
        return self._check_attachment_points(mol) and self._verify_conditions(mol)

    def _check_attachment_points(self, mol: Mol) -> bool:
        check = [
            atom.GetDegree() == 1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN
        ]
        return all(check) and len(check) > 0

    def _verify_conditions(self, mol: Mol) -> bool:
        try:
            verification = []
            for condition in self.conditions:
                descriptor = self._CONDITIONS_FUNC.get(condition.name)
                if condition.equals:
                    verification.append(condition.equals == descriptor(mol))
                if condition.min:
                    verification.append(condition.min < descriptor(mol))
                if condition.max:
                    verification.append(condition.max > descriptor(mol))
                if not all(verification):
                    return False
        except:
            return False
        return True
