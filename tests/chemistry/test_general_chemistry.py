import unittest

from rdkit import Chem
from reinvent.chemistry import conversions
from tests.chemistry.fixtures.test_data import (
    INVALID,
    ASPIRIN2,
    METHYL_3_O_TOLYL_PROPYL_AMINE,
    METHYL_3_O_TOLYL_PROPYL_AMINE2,
    CELECOXIB2,
    ASPIRIN_INCHI_KEY,
    CELECOXIB_INCHI_KEY,
)


class Test_general_chemistry(unittest.TestCase):
    def setUp(self):
        self.smiles = [ASPIRIN2, CELECOXIB2, INVALID]
        self.stereo_smiles = METHYL_3_O_TOLYL_PROPYL_AMINE2
        self.non_stereo_smiles = METHYL_3_O_TOLYL_PROPYL_AMINE
        self.mols = [Chem.MolFromSmiles(smile) for smile in [CELECOXIB2, ASPIRIN2]]

    def test_smiles_to_mols_and_indices(self):
        mols, indices = conversions.smiles_to_mols_and_indices(self.smiles)

        self.assertEqual(len(mols), 2)
        self.assertEqual(len(indices), 2)

    def test_mols_to_fingerprints(self):
        fps = conversions.mols_to_fingerprints(self.mols)

        self.assertEqual(len(fps), 2)

    def test_smiles_to_mols(self):
        mols = conversions.smiles_to_mols(self.smiles)

        self.assertEqual(len(mols), 2)

    def test_smiles_to_fingerprints(self):
        fps = conversions.smiles_to_fingerprints(self.smiles)

        self.assertEqual(len(fps), 2)

    def test_smile_to_mol_not_none(self):
        mol = conversions.smile_to_mol(ASPIRIN2)

        self.assertIsNotNone(mol)

    def test_smile_to_mol_none(self):
        mol = conversions.smile_to_mol(INVALID)

        self.assertIsNone(mol)

    def test_mols_to_smiles(self):
        mols = conversions.smiles_to_mols(self.smiles)
        smiles = conversions.mols_to_smiles(mols)

        self.assertEqual(self.smiles[:2], smiles)

    def test_mols_to_smiles_stereo(self):
        mols = conversions.smile_to_mol(self.stereo_smiles)
        smiles = conversions.mol_to_smiles(mols)

        self.assertEqual(self.non_stereo_smiles, smiles)
