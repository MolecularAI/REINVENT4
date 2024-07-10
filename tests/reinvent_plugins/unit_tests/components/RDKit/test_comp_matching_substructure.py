import pytest

from reinvent_plugins.components.RDKit.comp_matching_substructure import (
    Parameters,
    MatchingSubstructure,
)


@pytest.mark.parametrize("use_chirality", [True, False])
def test_comp_matching_substructure(use_chirality):
    smarts = ["ClccccF"]
    smiles = ["c1cc(F)ccc1Cl"]
    params = Parameters(smarts, [use_chirality])
    ms = MatchingSubstructure(params)
    results = ms(smiles)
    assert results.scores[0] == 1.0
