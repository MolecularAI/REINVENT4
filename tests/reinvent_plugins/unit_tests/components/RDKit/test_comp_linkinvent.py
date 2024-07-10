import numpy as np

from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentQed
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentMolecularWeight
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentTPSA
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentDistanceMatrix
from reinvent_plugins.components.RDKit.comp_linkinvent import (
    FragmentNumAtomStereoCenters,
)
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentHBondAcceptors
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentHBondDonors
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentNumRotBond
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentCsp3
from reinvent_plugins.components.RDKit.comp_linkinvent import (
    Fragmentnumsp,
    Fragmentnumsp2,
    Fragmentnumsp3,
)
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentNumHeavyAtoms
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentNumHeteroAtoms
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentNumRings
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentNumAromaticRings
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentNumAliphaticRings
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentSlogP
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentEffectiveLength
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentGraphLength
from reinvent_plugins.components.RDKit.comp_linkinvent import FragmentLengthRatio


def test_comp_linkinvent():
    input_fragments = ["[*]c1ccc2c(N)noc2c1[*]"]

    expected_results = {
        FragmentQed: np.array([0.592209977]),
        FragmentMolecularWeight: np.array([134.138]),
        FragmentTPSA: np.array([52.05]),
        FragmentDistanceMatrix: np.array(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 3.0, 2.0],
                    [2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0],
                    [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
                    [4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 3.0, 3.0, 4.0],
                    [4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0],
                    [3.0, 4.0, 3.0, 2.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0],
                    [2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0],
                    [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                ]
            ]
        ),
        FragmentNumAtomStereoCenters: np.array([0.0]),
        FragmentHBondAcceptors: np.array([3]),
        FragmentHBondDonors: np.array([1]),
        FragmentNumRotBond: np.array([0.0]),
        FragmentCsp3: np.array([0.0]),
        Fragmentnumsp: np.array([0.0]),
        Fragmentnumsp2: np.array([10.0]),
        Fragmentnumsp3: np.array([0.0]),
        FragmentNumHeavyAtoms: np.array([10.0]),
        FragmentNumHeteroAtoms: np.array([3.0]),
        FragmentNumRings: np.array([2.0]),
        FragmentNumAromaticRings: np.array([2.0]),
        FragmentNumAliphaticRings: np.array([0.0]),
        FragmentSlogP: np.array([1.41]),
        FragmentEffectiveLength: np.array([1]),
        FragmentGraphLength: np.array([5]),
        FragmentLengthRatio: np.array([20]),
    }

    for component in expected_results:
        results = component()(input_fragments)
        assert np.allclose(results.scores[0], expected_results[component])


def test_linker_effective_length():
    # 4 attachment points
    input_fragments = ["[*]c1ccc(N(c2ccc3c(c2)C([*])([*])c2ccccc2-3)[*])cc1"]
    expected_results = {
        FragmentEffectiveLength: np.array([0]),
        FragmentGraphLength: np.array([11]),
        FragmentLengthRatio: np.array([0]),
    }

    for component in expected_results:
        results = component()(input_fragments)
        assert np.allclose(results.scores[0], expected_results[component])


def test_linker_effective_length_ratio():
    # linker with single atom
    input_fragments = ["[*]N[*]"]
    expected_results = {
        FragmentEffectiveLength: np.array([0]),
        FragmentGraphLength: np.array([0]),
        FragmentLengthRatio: np.array([1]),
    }

    for component in expected_results:
        results = component()(input_fragments)
        assert np.allclose(results.scores[0], expected_results[component])
