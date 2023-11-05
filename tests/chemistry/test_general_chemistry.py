import unittest

from rdkit import Chem
from reinvent.chemistry import Conversions
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
        self.chemistry = Conversions()
        self.smiles = [ASPIRIN2, CELECOXIB2, INVALID]
        self.stereo_smiles = METHYL_3_O_TOLYL_PROPYL_AMINE2
        self.non_stereo_smiles = METHYL_3_O_TOLYL_PROPYL_AMINE
        self.mols = [Chem.MolFromSmiles(smile) for smile in [CELECOXIB2, ASPIRIN2]]
        self.inchi_keys = [ASPIRIN_INCHI_KEY, CELECOXIB_INCHI_KEY]

    def test_smiles_to_mols_and_indices(self):
        mols, indices = self.chemistry.smiles_to_mols_and_indices(self.smiles)

        self.assertEqual(len(mols), 2)
        self.assertEqual(len(indices), 2)

    def test_mols_to_fingerprints(self):
        fps = self.chemistry.mols_to_fingerprints(self.mols)

        self.assertEqual(len(fps), 2)

    def test_smiles_to_mols(self):
        mols = self.chemistry.smiles_to_mols(self.smiles)

        self.assertEqual(len(mols), 2)

    def test_smiles_to_fingerprints(self):
        fps = self.chemistry.smiles_to_fingerprints(self.smiles)

        self.assertEqual(len(fps), 2)

    def test_smile_to_mol_not_none(self):
        mol = self.chemistry.smile_to_mol(ASPIRIN2)

        self.assertIsNotNone(mol)

    def test_smile_to_mol_none(self):
        mol = self.chemistry.smile_to_mol(INVALID)

        self.assertIsNone(mol)

    def test_mols_to_smiles(self):
        mols = self.chemistry.smiles_to_mols(self.smiles)
        smiles = self.chemistry.mols_to_smiles(mols)

        self.assertEqual(self.smiles[:2], smiles)

    def test_mols_to_smiles_stereo(self):
        mols = self.chemistry.smile_to_mol(self.stereo_smiles)
        smiles = self.chemistry.mol_to_smiles(mols)

        self.assertEqual(self.non_stereo_smiles, smiles)

    def test_mol_to_inchi_key(self):
        inchi_keys = [self.chemistry.mol_to_inchi_key(mol) for mol in self.mols]
        self.assertEqual(self.inchi_keys, inchi_keys)
