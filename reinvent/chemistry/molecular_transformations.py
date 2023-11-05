import re
from typing import List

from rdkit.Chem.rdchem import Mol, RWMol, BondType
from rdkit.Chem.rdmolops import CombineMols, SanitizeMol
from reinvent.chemistry.conversions import Conversions

from reinvent.chemistry import TransformationTokens


class MolecularTransformations:
    def __init__(self):
        self._conversions = Conversions()
        self._tokens = TransformationTokens()

    def join_scaffolds_and_decorations(self, scaffold_smi, decorations_smi, canonicalize=True):
        decorations_smi = [
            self.add_first_attachment_point_number(dec, i)
            for i, dec in enumerate(decorations_smi.split(self._tokens.ATTACHMENT_SEPARATOR_TOKEN))
        ]
        scaffold_smi = self.add_attachment_point_numbers(scaffold_smi, canonicalize)
        num_attachment_points = len(self.get_attachment_points(scaffold_smi))
        if len(decorations_smi) != num_attachment_points:
            return None

        mol = self._conversions.smile_to_mol(scaffold_smi)
        for decoration in decorations_smi:
            mol = self.join_molecule_fragments(mol, self._conversions.smile_to_mol(decoration))
            if not mol:
                return None
        return mol

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

    def get_attachment_points(self, mol_or_smi) -> List:
        """
        Gets all attachment points regardless of the format.
        :param mol_or_smi: A Mol object or a SMILES string
        :return : A list with the numbers ordered by appearance.
        """
        if isinstance(mol_or_smi, Mol):
            return [
                int(atom.GetProp("molAtomMapNumber"))
                for atom in mol_or_smi.GetAtoms()
                if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN
                and atom.HasProp("molAtomMapNumber")
            ]
        return [
            int(match.group(1))
            for match in re.finditer(self._tokens.ATTACHMENT_POINT_NUM_REGEXP, mol_or_smi)
        ]

    def join_molecule_fragments(self, scaffold, decoration, keep_label_on_atoms=False):
        """
        Joins a RDKit MOL scaffold with a decoration. They must be labelled.
        :param scaffold_smi: RDKit MOL of the scaffold.
        :param decoration_smi: RDKit MOL of the decoration.
        :param keep_label_on_atoms: Add the labels to the atoms after attaching the molecule.
        This is useful when debugging, but it can give problems.
        :return: A Mol object of the joined scaffold.
        """

        if scaffold and decoration:
            # obtain id in the decoration
            try:
                attachment_points = [
                    atom.GetProp("molAtomMapNumber")
                    for atom in decoration.GetAtoms()
                    if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN
                ]
                if len(attachment_points) != 1:
                    return None  # more than one attachment point...
                attachment_point = attachment_points[0]
            except KeyError:
                return None

            combined_scaffold = RWMol(CombineMols(decoration, scaffold))
            attachments = [
                atom
                for atom in combined_scaffold.GetAtoms()
                if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN
                and atom.HasProp("molAtomMapNumber")
                and atom.GetProp("molAtomMapNumber") == attachment_point
            ]
            if len(attachments) != 2:
                return None  # something weird

            neighbors = []
            for atom in attachments:
                if atom.GetDegree() != 1:
                    return None  # the attachment is wrongly generated
                neighbors.append(atom.GetNeighbors()[0])

            bonds = [atom.GetBonds()[0] for atom in attachments]
            bond_type = BondType.SINGLE
            if any(bond for bond in bonds if bond.GetBondType() == BondType.DOUBLE):
                bond_type = BondType.DOUBLE

            combined_scaffold.AddBond(neighbors[0].GetIdx(), neighbors[1].GetIdx(), bond_type)
            combined_scaffold.RemoveAtom(attachments[0].GetIdx())
            combined_scaffold.RemoveAtom(attachments[1].GetIdx())

            if keep_label_on_atoms:
                for neigh in neighbors:
                    self._add_attachment_point_num(neigh, attachment_point)

            # Label the atoms in the bond
            bondNumbers = [
                int(atom.GetProp("bondNum"))
                for atom in combined_scaffold.GetAtoms()
                if atom.HasProp("bondNum")
            ]

            if bondNumbers:
                bondNum = max(bondNumbers) + 1
            else:
                bondNum = 0

            for neighbor in neighbors:
                idx = neighbor.GetIdx()
                atom = combined_scaffold.GetAtomWithIdx(idx)
                atom.SetIntProp("bondNum", bondNum)
            ##########################################

            scaffold = combined_scaffold.GetMol()
            try:
                SanitizeMol(scaffold)
            except ValueError:  # sanitization error
                return None
        else:
            return None

        return scaffold

    def _add_attachment_point_num(self, atom, idx):
        idxs = []
        if atom.HasProp("molAtomMapNumber"):
            idxs = atom.GetProp("molAtomMapNumber").split(",")
        idxs.append(str(idx))
        idxs = sorted(list(set(idxs)))
        atom.SetProp("molAtomMapNumber", ",".join(idxs))

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

    def randomize_scaffold(self, scaffold: Mol):
        smi = self._conversions.mol_to_random_smiles(scaffold)
        conv_smi = None
        if smi:
            conv_smi = self._add_brackets_to_attachment_points(smi)
        return conv_smi

    def _add_brackets_to_attachment_points(self, scaffold: str):
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

    def get_first_attachment_point(self, mol_or_smi):
        """
        Obtains the number of the first attachment point.
        :param mol_or_smi: A Mol object or a SMILES string
        :return: The number of the first attachment point.
        """
        return self.get_attachment_points(mol_or_smi)[0]
