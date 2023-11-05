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
