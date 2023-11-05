from typing import Optional

from rdkit.Chem.rdchem import Mol, BondType, RWMol
from rdkit.Chem.rdmolops import SanitizeMol, CombineMols

from reinvent.chemistry import Conversions, TransformationTokens
from reinvent.chemistry.library_design import AttachmentPoints


class BondMaker:
    def __init__(self):
        self._conversions = Conversions()
        self._tokens = TransformationTokens()
        self._attachment_points = AttachmentPoints()

    def join_scaffolds_and_decorations(
        self, scaffold_smi: str, decorations_smi, keep_labels_on_atoms=False
    ) -> Optional[Mol]:
        decorations_smi = [
            self._attachment_points.add_first_attachment_point_number(dec, i)
            for i, dec in enumerate(decorations_smi.split(self._tokens.ATTACHMENT_SEPARATOR_TOKEN))
        ]
        num_attachment_points = len(self._attachment_points.get_attachment_points(scaffold_smi))
        if len(decorations_smi) != num_attachment_points:
            return None

        mol = self._conversions.smile_to_mol(scaffold_smi)
        for decoration in decorations_smi:
            mol = self.join_molecule_fragments(
                mol,
                self._conversions.smile_to_mol(decoration),
                keep_label_on_atoms=keep_labels_on_atoms,
            )
            if not mol:
                return None
        return mol

    def join_molecule_fragments(self, scaffold: Mol, decoration: Mol, keep_label_on_atoms=False):
        """
        Joins a RDKit MOL scaffold with a decoration. They must be labelled.
        :param scaffold: RDKit MOL of the scaffold.
        :param decoration: RDKit MOL of the decoration.
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
        # Fixme: This way of annotating fails in case of several attachment points when the mol is converted back to a
        #  SMILES string (RuntimeError: boost::bad_any_cast: failed conversion using boost::any_cast)
        #  For example combining scaffold '*C(*)CC' and  warhead pair '*OC|*C' would result in
        #  C[O:0][CH:0,1]([CH3:1])CC, which results in an error due to the '0,1'

    def randomize_scaffold(self, scaffold: Mol):
        smi = self._conversions.mol_to_random_smiles(scaffold)
        conv_smi = None
        if smi:
            conv_smi = self._attachment_points.add_brackets_to_attachment_points(smi)
        return conv_smi


################################################################


# def join_two_fragments(self, scaffold, decoration):
#     combined_scaffold = None
#
#     if scaffold and decoration:
#         decoration_attachment = self.get_attachment_point_for_decoration(decoration)
#         scaff_attachment = self.get_attachment_points_for_scaffold(scaffold, decoration_attachment.GetProp(
#             "molAtomMapNumber"))
#         # neighbors = self.get_attachment_point_neighboring_atoms(scaff_attachment, decoration_attachment)
#         combined_scaffold = RWMol(CombineMols(scaffold, decoration))
#         attachments = self.get_attachment_points_for_ensemble(combined_scaffold, decoration_attachment.GetProp("molAtomMapNumber"))
#         neighbors = self.get_attachment_point_neighboring_atoms(attachments[1], attachments[0])
#
#         if len(neighbors) == 2:
#             bond_pair = self._get_indices_from_atom_collection(neighbors)
#             attachment_indices = self._get_indices_from_atom_collection([scaff_attachment, decoration_attachment])
#             combined_scaffold = self.form_bonds(combined_scaffold, bond_pair)
#             combined_scaffold = self.remove_attachment_points(combined_scaffold, attachment_indices)
#
#     return combined_scaffold
#
# def _update_molecule(self, molecule: Mol):
#     scaffold = molecule.GetMol()
#     try:
#         SanitizeMol(scaffold)
#     except ValueError:  # sanitization error
#         return None
#     return scaffold
#
# def get_attachment_point_for_decoration(self, decoration: Mol) -> Atom:
#     attachment_point = None
#     try:
#         attachment_points = [atom for atom in decoration.GetAtoms()
#                              if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN]
#         if len(attachment_points) == 1:
#             attachment_point = attachment_points[0]
#             # print(type(attachment_point))
#     except KeyError:
#         pass
#     return attachment_point
#
# def get_attachment_points_for_scaffold(self, scaffold: Mol, attachment_point: str) -> Atom:
#     attachments = [atom for atom in scaffold.GetAtoms()
#                    if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN and
#                    atom.HasProp("molAtomMapNumber") and atom.GetProp("molAtomMapNumber") == attachment_point]
#     ############
#     print("+"*80)
#     for a in attachments:
#         print(f'get_attachment_points_for_scaffold {a.GetProp("molAtomMapNumber")} {a.GetIdx()} symbol {a.GetSymbol()}')
#     ###########
#     if len(attachments) != 1:
#         return None
#     attachment = attachments[0]
#     return attachment
#
# def get_attachment_points_for_ensemble(self, combined_scaffold: Mol, attachment_point: str) -> List[Atom]:
#     attachments = [atom for atom in combined_scaffold.GetAtoms()
#                    if atom.GetSymbol() == self._tokens.ATTACHMENT_POINT_TOKEN and
#                    atom.HasProp("molAtomMapNumber") and atom.GetProp("molAtomMapNumber") == attachment_point]
#     print("+"*80)
#     for a in attachments:
#         print(f'{a.GetProp("molAtomMapNumber")} {a.GetIdx()}')
#
#     if len(attachments) != 2:
#         attachments = []  # something weird
#     return attachments
#
# def get_attachment_point_neighboring_atoms(self, scaffold_attachment, decoration_point) -> List[Atom]:
#     neighbors = []
#     if scaffold_attachment.GetDegree() == 1 and decoration_point.GetDegree() == 1:
#         neighbors = [scaffold_attachment.GetNeighbors()[0], decoration_point.GetNeighbors()[0]]
#     return neighbors
#
# def remove_attachment_points(self, combined_scaffold: Mol, attachments: List[int]) -> Mol:
#     for index in sorted(attachments, reverse=True):
#         combined_scaffold.RemoveAtom(index)
#     return combined_scaffold
#
# def form_bonds(self, combined_scaffold: Mol, bond_pairs: List[int]) -> Mol:
#     combined_scaffold.AddBond(bond_pairs[0], bond_pairs[1], BondType.SINGLE)
#     return combined_scaffold
#
# def _get_indices_from_atom_collection(self, collection: List[Atom]) -> List[int]:
#     indices = [atom.GetIdx() for atom in collection]
#     return indices
