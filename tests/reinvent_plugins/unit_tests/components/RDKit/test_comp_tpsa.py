import numpy as np

from reinvent_plugins.components.RDKit.comp_tpsa import TPSA, Parameters


smilies = (
    "Cc1[nH]c(=O)[nH]c(=S)c1C(=O)NC(C)(CO)CO",
    "CC(c1cccc(F)n1)c1c(CCN(C)C)sc2ccccc12",
    "N=c1[nH]cnc2c1cnn2CCC(=O)O",
)


def test_comp_tpsa():
    expected_results = [np.array([118.21, 16.13, 107.65]), np.array([150.3, 44.37, 107.65])]

    params = Parameters([False, True])
    component = TPSA(params)

    results = component(smilies)

    assert np.allclose(results.scores, expected_results)


def test_comp_tpsa_default_all_false():
    expected_results = [np.array([118.21, 16.13, 107.65]), np.array([118.21, 16.13, 107.65])]

    params = Parameters()
    component = TPSA(params)

    results = component(smilies)

    assert np.allclose(results.scores, expected_results)
