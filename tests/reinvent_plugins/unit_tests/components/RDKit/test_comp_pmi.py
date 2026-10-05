"""Test for the RDKit PMI descriptor

NOTE: This is a 3D descriptor and therefore the scoring component creates
      a conformer.  However, this is not deterministic meaning that the
      generated 3D coordinates can vary depending on the flexibility of the
      molecule and the quality of the conformer generator.  The examples
      here are basically rigid (except for a rotating methyl group).

      This also means that the test cases here are not really unit tests.
"""

import numpy as np
import pytest

from reinvent_plugins.components.RDKit.comp_pmi import Parameters, PMI

RTOL = 0.05  # this makes the tests fairly fragile


def test_comp_pmi_all():
    smiles = ["c1ccccc1", "Cc1ccccc1"]
    expected_results = [np.array([0.487, 0.312]), np.array([0.513, 0.700])]
    expected_results_rdkit_2025 = [np.array([0.494, 0.333]), np.array([0.506, 0.679])]

    params = Parameters(["npr1", "npr2"])
    pmi = PMI(params)
    results = pmi(smiles)

    assert np.allclose(results.scores, expected_results, rtol=RTOL) or np.allclose(
        results.scores, expected_results_rdkit_2025, rtol=RTOL
    )


def test_comp_pmi_npr1():
    smiles = ["c1ccccc1", "Cc1ccccc1"]
    expected_results = np.array([0.487, 0.312])

    params = Parameters(["npr1"])
    pmi = PMI(params)
    results = pmi(smiles)

    assert np.allclose(results.scores, expected_results, rtol=RTOL)


def test_comp_pmi_npr2():
    smiles = ["c1ccccc1", "Cc1ccccc1"]
    expected_results = np.array([0.513, 0.700])

    params = Parameters(["npr2"])
    pmi = PMI(params)
    results = pmi(smiles)

    assert np.allclose(results.scores, expected_results, rtol=RTOL)


def test_comp_pmi_endpoint_order():
    # a linear molecule has NPR1 ~ 0 and NPR2 ~ 1 for any conformer
    smiles = ["C#CC#C"]

    params = Parameters(["npr2", "npr1"])
    pmi = PMI(params)
    results = pmi(smiles)

    assert len(results.scores) == pmi.number_of_endpoints == 2
    assert np.allclose(results.scores[0], [1.0], atol=0.01)  # npr2
    assert np.allclose(results.scores[1], [0.0], atol=0.01)  # npr1


def test_comp_pmi_repeated_property():
    smiles = ["C#CC#C"]

    params = Parameters(["npr1", "npr1"])
    pmi = PMI(params)
    results = pmi(smiles)

    assert len(results.scores) == pmi.number_of_endpoints == 2


def test_comp_pmi_unknown_property():
    params = Parameters(["npr1", "npr3"])

    with pytest.raises(ValueError):
        PMI(params)
