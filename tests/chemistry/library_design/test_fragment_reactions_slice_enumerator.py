import unittest

from reinvent.chemistry import Conversions
from reinvent.chemistry.library_design import (
    FragmentReactionSliceEnumerator,
    BondMaker,
    AttachmentPoints,
)
from reinvent.chemistry.library_design.dtos import FilteringConditionDTO
from reinvent.chemistry.library_design.enums import MolecularDescriptorsEnum
from reinvent.chemistry.library_design.fragment_reactions import FragmentReactions
from tests.chemistry.library_design.fixtures import FRAGMENT_REACTION_SUZUKI, FRAGMENT_REACTIONS
from tests.chemistry.fixtures.test_data import CELECOXIB


class TestSingleFragmentReactionsSliceEnumerator(unittest.TestCase):
    def setUp(self):
        self.chemistry = Conversions()
        self.reactions = FragmentReactions()
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._suzuki_reaction_dto_list = self.reactions.create_reactions_from_smirks(
            FRAGMENT_REACTION_SUZUKI
        )
        self.suzuki_positive_smile = CELECOXIB
        self.suzuki_positive_molecule = self.chemistry.smile_to_mol(self.suzuki_positive_smile)

        scaffold_conditions = []
        decoration_conditions = []
        self._slice_enumerator = FragmentReactionSliceEnumerator(
            self._suzuki_reaction_dto_list, scaffold_conditions, decoration_conditions
        )

    def test_enumeration_slcing_1(self):
        result = self._slice_enumerator.enumerate(self.suzuki_positive_molecule, 1)

        self.assertEqual(2, len(result))

    def test_enumeration_slicing_2(self):
        result = self._slice_enumerator.enumerate(self.suzuki_positive_molecule, 4)

        self.assertEqual(2, len(result))

        complete_molecules = []
        for sliced in result:
            values = [
                self.chemistry.mol_to_smiles(smi) for num, smi in sorted(sliced.decorations.items())
            ]
            decorations = "|".join(values)
            labeled_scaffold = self._attachment_points.add_attachment_point_numbers(sliced.scaffold)
            molecule = self._bond_maker.join_scaffolds_and_decorations(
                labeled_scaffold, decorations
            )
            complete_smile = self.chemistry.mol_to_smiles(molecule)
            complete_molecules.append(complete_smile)
        self.assertEqual(
            self.chemistry.mol_to_smiles(self.suzuki_positive_molecule),
            list(set(complete_molecules))[0],
        )


class TestMultipleFragmentReactionsSliceEnumerator(unittest.TestCase):
    def setUp(self):
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self.chemistry = Conversions()
        self.reactions = FragmentReactions()
        self._fragment_reactions = self.reactions.create_reactions_from_smirks(FRAGMENT_REACTIONS)
        self.positive_smile = CELECOXIB

        self.positive_molecule = self.chemistry.smile_to_mol(self.positive_smile)

        scaffold_conditions = []
        decoartion_conditions = []
        self._slice_enumerator = FragmentReactionSliceEnumerator(
            self._fragment_reactions, scaffold_conditions, decoartion_conditions
        )

    def test_enumeration_slicing_1(self):
        result = self._slice_enumerator.enumerate(self.positive_molecule, 1)

        complete_molecules = []
        for sliced in result:
            values = [
                self.chemistry.mol_to_smiles(smi) for num, smi in sorted(sliced.decorations.items())
            ]
            decorations = "|".join(values)
            labeled_scaffold = self._attachment_points.add_attachment_point_numbers(sliced.scaffold)
            molecule = self._bond_maker.join_scaffolds_and_decorations(
                labeled_scaffold, decorations
            )
            complete_smile = self.chemistry.mol_to_smiles(molecule)
            complete_molecules.append(complete_smile)
        reference = self.chemistry.mol_to_smiles(self.chemistry.smile_to_mol(self.positive_smile))
        self.assertEqual(reference, list(set(complete_molecules))[0])

        self.assertEqual(6, len(result))

    def test_enumeration_slicing_2(self):
        result = self._slice_enumerator.enumerate(self.positive_molecule, 2)
        self.assertEqual(12, len(result))

    def test_enumeration_slicing_3(self):
        result = self._slice_enumerator.enumerate(self.positive_molecule, 3)

        complete_molecules = []
        for sliced in result:
            values = [self.chemistry.mol_to_smiles(smi) for num, smi in sliced.decorations.items()]
            decorations = "|".join(values)
            labeled_scaffold = self._attachment_points.add_attachment_point_numbers(sliced.scaffold)
            molecule = self._bond_maker.join_scaffolds_and_decorations(
                labeled_scaffold, decorations
            )
            complete_smile = self.chemistry.mol_to_smiles(molecule)
            complete_molecules.append(complete_smile)
        reference = self.chemistry.mol_to_smiles(self.chemistry.smile_to_mol(self.positive_smile))
        self.assertEqual(reference, list(set(complete_molecules))[0])
        self.assertEqual(1, len(list(set(complete_molecules))))
        self.assertEqual(12, len(result))


class TestReactionsSliceEnumeratorWithFilters(unittest.TestCase):
    def setUp(self):
        self.chemistry = Conversions()
        self.reactions = FragmentReactions()
        self._suzuki_reactions = self.reactions.create_reactions_from_smirks(
            FRAGMENT_REACTION_SUZUKI
        )
        self.positive_smile = CELECOXIB

        self.positive_molecule = self.chemistry.smile_to_mol(self.positive_smile)
        descriptors_enum = MolecularDescriptorsEnum()

        scaffold_condition_1 = FilteringConditionDTO(descriptors_enum.RING_COUNT, min=1)
        scaffold_condition_2 = FilteringConditionDTO(descriptors_enum.MOLECULAR_WEIGHT, max=600)
        decoartion_condition_1 = FilteringConditionDTO(
            descriptors_enum.HYDROGEN_BOND_ACCEPTORS, max=3
        )
        decoartion_condition_2 = FilteringConditionDTO(descriptors_enum.HYDROGEN_BOND_DONORS, max=3)
        scaffold_conditions = [scaffold_condition_1, scaffold_condition_2]
        decoartion_conditions = [decoartion_condition_1, decoartion_condition_2]
        self._slice_enumerator = FragmentReactionSliceEnumerator(
            self._suzuki_reactions, scaffold_conditions, decoartion_conditions
        )

    def test_enumeration_slicing_1(self):
        result = self._slice_enumerator.enumerate(self.positive_molecule, 1)

        self.assertEqual(1, len(result))
        self.assertEqual(result[0].original_smiles, result[0].reassembled_smiles)

    def test_enumeration_slicing_2(self):
        result = self._slice_enumerator.enumerate(self.positive_molecule, 2)

        self.assertEqual(1, len(result))
        self.assertEqual(result[0].original_smiles, result[0].reassembled_smiles)
