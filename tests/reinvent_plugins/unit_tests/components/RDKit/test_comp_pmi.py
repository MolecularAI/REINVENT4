import numpy as np

from reinvent_plugins.components.RDKit.comp_pmi import Parameters, PMI


def test_comp_pmi():
    params = Parameters(["npr1", "npr2"])
    smiles = ["c1ccccc1", "Cc1ccccc1"]
    pmi = PMI(params)
    results = pmi(smiles)
    expected_results = np.array([0.48749534, 0.31194547, 0.51250466, 0.69953459])
    assert np.allclose(np.concatenate(results.scores), expected_results)
