from collections import OrderedDict

from rdkit.Chem.rdchem import Mol

from reinvent.chemistry import Conversions, TransformationTokens
from reinvent.chemistry.library_design import BondMaker, AttachmentPoints


class FragmentedMolecule:
    def __init__(self, scaffold: Mol, decorations: OrderedDict, original_smiles: str):
        """
        Represents a molecule as a scaffold and the decorations associated with each attachment point.
        :param scaffold: A Mol object with the scaffold.
        :param decorations: Either a list or a dict with the decorations as Mol objects.
        """
        self._tockens = TransformationTokens()
        self._attachments = AttachmentPoints()
        self._conversions = Conversions()
        self._bond_maker = BondMaker()
        self.scaffold = scaffold
        self.decorations = decorations
        self.original_smiles = original_smiles
        self.scaffold_smiles = self._conversions.mol_to_smiles(self.scaffold)
        self.re_label()
        self.decorations_smiles = self._create_decorations_string()
        self.reassembled_smiles = self._re_assemble()
        # self.reorder_attachment_point_numbers()

    def __eq__(self, other):
        return (
            self.decorations_smiles == other.decorations_smiles
            and self.scaffold_smiles == other.scaffold_smiles
        )

    def __hash__(self):
        return tuple([self.scaffold_smiles, self.decorations_smiles]).__hash__()

    def decorations_count(self) -> int:
        return len(self.decorations)

    def re_label(self):
        labels = self._attachments.get_attachment_points(self.scaffold_smiles)
        decorations = OrderedDict()
        for i, v in enumerate(labels):
            decorations[i] = self.decorations[v]
        self.decorations = decorations

    def reorder_attachment_point_numbers(self):
        self.scaffold_smiles = self._attachments.add_attachment_point_numbers(self.scaffold_smiles)

    def _re_assemble(self):
        self.scaffold_smiles = self._attachments.add_attachment_point_numbers(
            self.scaffold_smiles, canonicalize=False
        )
        molecule = self._bond_maker.join_scaffolds_and_decorations(
            self.scaffold_smiles, self.decorations_smiles
        )
        return self._conversions.mol_to_smiles(molecule)

    def _create_decorations_string(self):
        values = [self._conversions.mol_to_smiles(smi) for num, smi in self.decorations.items()]
        decorations = "|".join(values)
        return decorations

    def to_smiles(self):
        """
        Calculates the SMILES representation of the given variant of the scaffold and decorations.
        :param variant: SMILES variant to use (see to_smiles)
        :return: A tuple with the SMILES of the scaffold and a dict with the SMILES of the decorations.
        """
        return (
            self._conversions.mol_to_smiles(self.scaffold),
            {num: self._conversions.mol_to_smiles(dec) for num, dec in self.decorations.items()},
        )

    def to_random_smiles(self):
        """
        Calculates the SMILES representation of the given variant of the scaffold and decorations.
        :param variant: SMILES variant to use (see to_smiles)
        :return: A tuple with the SMILES of the scaffold and a dict with the SMILES of the decorations.
        """
        return (
            self._conversions.mol_to_random_smiles(self.scaffold),
            {
                num: self._conversions.mol_to_random_smiles(dec)
                for num, dec in self.decorations.items()
            },
        )
