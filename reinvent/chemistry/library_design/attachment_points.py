import re
from typing import List

from rdkit.Chem.rdchem import Mol

from reinvent.chemistry import Conversions, TransformationTokens


class AttachmentPoints:
    def __init__(self):
        self._conversions = Conversions()
        self._tokens = TransformationTokens()

    def add_attachment_point_numbers(self, mol_or_smi, canonicalize=True):
        """
        Adds the numbers for the attachment points throughout the molecule.
        :param mol_or_smi: SMILES string to convert.
        :param canonicalize: Canonicalize the SMILES so that the attachment points are always in the same order.
        :return : A converted SMILES string.
        """
        if isinstance(mol_or_smi, str):
            smi = mol_or_smi
            if canonicalize:
                smi = self._conversions.mol_to_smiles(self._conversions.smile_to_mol(mol_or_smi))
            # only add numbers ordered by the SMILES ordering
            num = -1

            def _ap_callback(_):
                nonlocal num
                num += 1
                return "[{}:{}]".format(self._tokens.ATTACHMENT_POINT_TOKEN, num)

            return re.sub(self._tokens.ATTACHMENT_POINT_REGEXP, _ap_callback, smi)
        else:
            mol = mol_or_smi
            if canonicalize:
                mol = self._conversions.smile_to_mol(self._conversions.mol_to_smiles(mol))
            idx = 0
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN:
                    atom.SetProp("molAtomMapNumber", str(idx))
                    idx += 1
            return self._conversions.mol_to_smiles(mol)

    def get_attachment_points(self, smile: str) -> List:
        """
        Gets all attachment points from SMILES string.
        :param smile: A SMILES string
        :return : A list with the numbers ordered by appearance.
        """
        return [
            int(match.group(1))
            for match in re.finditer(self._tokens.ATTACHMENT_POINT_NUM_REGEXP, smile)
        ]

    def get_attachment_points_for_molecule(self, molecule: Mol) -> List:
        """
        Gets all attachment points from RDKit Mol.
        :param molecule: A Mol object.
        :return : A list with the numbers ordered by appearance.
        """
        if isinstance(molecule, Mol):
            return [
                int(atom.GetProp("molAtomMapNumber"))
                for atom in molecule.GetAtoms()
                if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN
                and atom.HasProp("molAtomMapNumber")
            ]

    def add_first_attachment_point_number(self, smi, num):
        """
        Changes/adds a number to the first attachment point.
        :param smi: SMILES string with the molecule.
        :param num: Number to add.
        :return: A SMILES string with the number added.
        """
        return re.sub(
            self._tokens.ATTACHMENT_POINT_REGEXP,
            "[{}:{}]".format(self._tokens.ATTACHMENT_POINT_TOKEN, num),
            smi,
            count=1,
        )

    def remove_attachment_point_numbers(self, smile: str) -> str:
        """
        Removes the numbers for the attachment points throughout the molecule.
        :param smile: SMILES string.
        :return : A converted SMILES string.
        """
        result = re.sub(
            self._tokens.ATTACHMENT_POINT_NUM_REGEXP,
            "[{}]".format(self._tokens.ATTACHMENT_POINT_TOKEN),
            smile,
        )
        return result

    def remove_attachment_point_numbers_from_mol(self, molecule: Mol) -> Mol:
        """
        Removes the numbers for the attachment points throughout the molecule.
        :param molecule: RDKit molecule.
        :return : A molecule.
        """
        if isinstance(molecule, Mol):
            for atom in molecule.GetAtoms():
                atom.ClearProp("molAtomMapNumber")
        return molecule

    def add_brackets_to_attachment_points(self, scaffold: str):
        """
        Adds brackets to the attachment points (if they don't have them).
        :param scaffold: SMILES string.
        :return: A SMILES string with attachments in brackets.
        """
        return re.sub(
            self._tokens.ATTACHMENT_POINT_NO_BRACKETS_REGEXP,
            "[{}]".format(self._tokens.ATTACHMENT_POINT_TOKEN),
            scaffold,
        )
