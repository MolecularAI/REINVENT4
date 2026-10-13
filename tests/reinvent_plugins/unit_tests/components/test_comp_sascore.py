import numpy as np
from reinvent_plugins.components.SAScore.comp_sascore import SAScore


def test_comp_sascore():
    inputs = [
        "O=C(C)Oc1ccccc1C(=O)O",
        "c1cc(ccc1C(=O)CCCN2CCC(CC2)(c3ccc(cc3)Cl)O)F",
        "C[C@@]1(C(=O)N2[C@H](C(=O)N3CCC[C@H]3[C@@]2(O1)O)CC4=CC=CC=C4)NC(=O)[C@H]5CN([C@@H]6CC7=CNC8=CC=CC(=C78)C6=C5)C",
    ]
    expected_values = [1.58003975, 2.12273261, 4.74082764]
    SAScore_scorer = SAScore()
    results = np.concatenate(SAScore_scorer(inputs).scores)

    assert np.allclose(results, expected_values)


def test_comp_sascore_none_molecule_scores_nan():
    # an invalid SMILES (None molecule from molcache) must score NaN rather
    # than raising (issue #333)
    component = SAScore()

    results = component(["not_a_molecule", "O=C(C)Oc1ccccc1C(=O)O"])

    assert np.isnan(results.scores[0][0])
    assert np.isclose(results.scores[0][1], 1.58003975)
