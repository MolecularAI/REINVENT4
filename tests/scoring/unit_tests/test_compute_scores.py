from dataclasses import dataclass
from typing import List, Dict, Optional

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

from reinvent.scoring.compute_scores import compute_component_scores, compute_transform
from reinvent.runmodes.samplers.sampler import validate_smiles
from reinvent.models.model_factory.sample_batch import SmilesState
from reinvent_plugins.components.component_results import ComponentResults,SmilesAssociatedComponentResults, SmilesResult


SMILIES = [
    "O=C(C)Oc1ccccc1C(=O)O",
    "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC",
    "CCC",
    "CCCCX",  # invalid
    "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC",  # duplicate
    "c1ccccc1c",  # invalid
    "c1cccc1N",  # invalid
    "C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl",
    "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # duplicate
]

# NOTE: validation also canonicalizes the SMILES
VALIDATED_SMILIES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C",
    "CCC",
    "CCCCX",
    "Cc1c(N(C)CS(=O)(=O)O)c(=O)n(-c2ccccc2)n1C",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C",
    "c1ccccc1c",
    "c1cccc1N",
    "Clc1ccc2c(c1)C(N1CCNCC1)=Nc1ccccc1O2",
    "CNC(C)C1CCC(N)C(OC2C(N)CC(N)C(OC3OCC(C)(O)C(NC)C3O)C2O)O1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
]

STATES = np.array(
    [
        SmilesState.VALID,
        SmilesState.VALID,
        SmilesState.VALID,
        SmilesState.VALID,
        SmilesState.VALID,
        SmilesState.INVALID,
        SmilesState.VALID,
        SmilesState.VALID,
        SmilesState.DUPLICATE,
        SmilesState.INVALID,
        SmilesState.INVALID,
        SmilesState.VALID,
        SmilesState.VALID,
        SmilesState.DUPLICATE,
    ]
)


def scoring_function(smilies):
    scores = []

    for smiles in smilies:
        try:
            mol = Chem.MolFromSmiles(smiles)
            score = Descriptors.MolWt(mol)
        except ValueError:
            score = np.nan

        scores.append(score)

    return ComponentResults([np.array(scores, dtype=float)])


# there is another test in runmodes
def test_validate_smiles():
    mols = [Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None for smiles in SMILIES]

    validated_smilies, states = validate_smiles(mols, SMILIES)

    assert validated_smilies == VALIDATED_SMILIES
    assert (states == STATES).all()


def test_compute_scores():
    mols = [Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None for smiles in SMILIES]

    validated_smilies, states = validate_smiles(mols, SMILIES)

    cache = {}
    invalid_mask = np.where(states == SmilesState.INVALID, False, True)
    duplicate_mask = np.where(states == SmilesState.DUPLICATE, False, True)

    component_results = compute_component_scores(
        validated_smilies, scoring_function, cache, invalid_mask & duplicate_mask
    )
    np.testing.assert_almost_equal(
        list(zip(*component_results.fetch_scores(validated_smilies)))[0],
        np.array(
            [
                180.159,
                381.379,
                206.285,
                303.358,
                44.097,
                0.0,
                311.363,
                194.194,
                303.358,
                0.0,
                0.0,
                313.788,
                477.603,
                206.285,
            ]
        ),
    )


def test_compute_scores_duplicates_in_cache():
    DUPLICATES = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    ]
    cache = {smiles: SmilesResult(score=(100.0,)) for smiles in DUPLICATES}
    mask = np.array([True, True, True])

    # the scoring function should never be called
    component_results = compute_component_scores(DUPLICATES, None, cache, mask)

    np.testing.assert_almost_equal(
        list(zip(*component_results.fetch_scores(DUPLICATES)))[0], np.array([100.0, 100.0, 100.0])
    )


def test_compute_scores_duplicates_not_in_cache():
    DUPLICATES = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    cache = {}
    mask = np.array([True, False, False])

    # the scoring function should eval fist component only
    component_results = compute_component_scores(DUPLICATES, scoring_function, cache, mask)

    np.testing.assert_almost_equal(
        list(zip(*component_results.fetch_scores(DUPLICATES)))[0],
        np.array(
            [
                180.159,
                180.159,
                180.159,
            ]
        ),
    )


def test_compute_component_scores_with_index_smiles_and_duplicates():
    smilies = ["CCO", "CCN", "CCO", "CCC"] # duplicates, e.g. same linkers in LinkInvent
    index_smiles = ["CCON", "CCNN", "CCOO", "CCCC"]  # unique, representing full molecules
    filter_mask = np.array([True, True, True, True])
    cache = {}

    component_results = compute_component_scores(
        smilies, scoring_function, cache, filter_mask, index_smiles=index_smiles
    )
    np.testing.assert_almost_equal(
        list(zip(*component_results.fetch_scores(index_smiles)))[0],
        np.array(
            [
                46.069,
                45.085,
                46.069,
                44.097,
            ]
        ),
    )

    assert set(component_results.data.keys()) == {"CCON", "CCNN", "CCOO", "CCCC"} # cehck all are scored