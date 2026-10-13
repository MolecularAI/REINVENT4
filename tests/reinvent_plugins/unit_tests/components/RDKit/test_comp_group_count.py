import numpy as np

from reinvent_plugins.components.RDKit.comp_group_count import (
    Parameters,
    GroupCount,
)


def test_comp_group_count():
    smarts = ["ClccccF"]
    smiles = ["c1cc(F)ccc1Cl"]
    params = Parameters(smarts)
    gc = GroupCount(params)
    results = gc(smiles)
    assert results.scores[0] == 2


def test_comp_group_count_none_molecule_scores_nan():
    # an invalid SMILES (None molecule from molcache) must score NaN rather
    # than raising (issue #333)
    params = Parameters(["ClccccF"])
    component = GroupCount(params)

    results = component(["not_a_molecule", "c1cc(F)ccc1Cl"])

    assert np.isnan(results.scores[0][0])
    assert results.scores[0][1] == 2
