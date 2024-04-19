import pytest

import numpy as np
from rdkit import Chem

from reinvent.runmodes.samplers import validate_smiles
from reinvent.models.model_factory.sample_batch import SmilesState
from tests import test_data


@pytest.fixture
def data():
    smilies = [
        test_data.PARACETAMOL,  # valid
        test_data.INVALID,  # invalid
        test_data.GENTAMICIN,  # valid
        test_data.PROPANE,  # valid
        test_data.CELECOXIB,  # valid
        test_data.NONSENSE,  # invalid
        test_data.GENTAMICIN,  # duplicate
        test_data.PARACETAMOL,  # duplicate
        "CC=O-C",  # invalid
        None,  # invalid
    ]

    yield smilies, [
        Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None for smiles in smilies
    ]

    pass  # no tear-down


def test_enough_smiles(data):
    smilies, mols = data

    validated_smilies, states = validate_smiles(mols, smilies)

    assert len(validated_smilies) == len(smilies)
    assert len(states) == len(smilies)


def test_number_of_valids(data):
    """Test for number of valid SMILES only"""
    smilies, mols = data

    validated_smilies, states = validate_smiles(mols, smilies)

    valid = np.where(states == SmilesState.VALID, True, False)
    assert sum(valid) == 4


def test_number_of_all_valids(data):
    """Test for number of valid and duplicate SMILES"""
    smilies, mols = data

    validated_smilies, states = validate_smiles(mols, smilies)

    all_valid = np.where(
        (states == SmilesState.VALID) | (states == SmilesState.DUPLICATE), True, False
    )
    assert sum(all_valid) == 6


def test_number_of_invalids(data):
    """Test for number of invalid SMILES"""
    smilies, mols = data

    validated_smilies, states = validate_smiles(mols, smilies)

    invalid = np.where(states == SmilesState.INVALID, True, False)
    assert sum(invalid) == 4
