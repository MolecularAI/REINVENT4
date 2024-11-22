from typing import Optional

from rdkit.Chem.rdchem import Mol, BondType, RWMol
from rdkit.Chem.rdmolops import SanitizeMol, CombineMols

from reinvent.chemistry import conversions, tokens
from reinvent.chemistry.library_design import attachment_points


def join_scaffolds_and_decorations(
    scaffold_smi: str, decorations_smi, keep_labels_on_atoms=False
) -> Optional[Mol]:
    decorations_smi = [
        attachment_points.add_first_attachment_point_number(dec, i)
        for i, dec in enumerate(decorations_smi.split(tokens.ATTACHMENT_SEPARATOR_TOKEN))
    ]
    num_attachment_points = len(attachment_points.get_attachment_points(scaffold_smi))
    if len(decorations_smi) != num_attachment_points:
        return None

    mol = conversions.smile_to_mol(scaffold_smi)
    for decoration in decorations_smi:
        mol = join_molecule_fragments(
            mol,
            conversions.smile_to_mol(decoration),
            keep_label_on_atoms=keep_labels_on_atoms,
        )
        if not mol:
            return None
    return mol


def join_molecule_fragments(scaffold: Mol, decoration: Mol, keep_label_on_atoms=False):
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
                if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN
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
            if atom.GetSymbol() == tokens.ATTACHMENT_POINT_TOKEN
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
                add_attachment_point_num(neigh, attachment_point)

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

        scaffold = combined_scaffold.GetMol()
        try:
            SanitizeMol(scaffold)
        except ValueError:  # sanitization error
            return None
    else:
        return None

    return scaffold


def add_attachment_point_num(atom, idx):
    idxs = []
    if atom.HasProp("molAtomMapNumber"):
        idxs = atom.GetProp("molAtomMapNumber").split(",")
    idxs.append(str(idx))
    idxs = sorted(list(set(idxs)))
    if len(idxs) > 1:
        idxs = idxs[0]  # keep the lowest index
    atom.SetProp("molAtomMapNumber", ",".join(idxs))
    # Fixme: This way of annotating fails in case of several attachment points when the mol is converted back to a
    #  SMILES string (RuntimeError: boost::bad_any_cast: failed conversion using boost::any_cast)
    #  For example combining scaffold '*C(*)CC' and  warhead pair '*OC|*C' would result in
    #  C[O:0][CH:0,1]([CH3:1])CC, which results in an error due to the '0,1'
    #  The heart of the issue is that molAtomMapNumber needs to be integer type, so we cannot store a list here.
    #  instead, keep the last attachement point only. This implies that reaction filters will not be compatible
    #  in the case of attachment points on the same atom (they did not work in R3 either)


def randomize_scaffold(scaffold: Mol):
    smi = conversions.mol_to_random_smiles(scaffold)
    conv_smi = None
    if smi:
        conv_smi = attachment_points.add_brackets_to_attachment_points(smi)
    return conv_smi
