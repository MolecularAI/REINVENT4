import pytest

from reinvent_plugins.components.RDKit.comp_matching_substructure import (
    Parameters,
    MatchingSubstructure,
)


def test_comp_matching_substructure():
    smarts = ["ClccccF"]
    smiles = ["c1cc(F)ccc1Cl"]
    params = Parameters(smarts, [False])
    ms = MatchingSubstructure(params)
    results = ms(smiles)
    assert results.scores[0][0] == 1.0


def test_comp_matching_substructure_list():
    smarts = [["CO", "CN"]]
    smiles = ["CCO", "CCN", "CCC"]
    params = Parameters(smarts, [False])
    ms = MatchingSubstructure(params)
    results = ms(smiles)
    print(results.scores)
    assert results.scores[0][0] == 1.0
    assert results.scores[0][1] == 1.0
    assert results.scores[0][2] == 0.5


@pytest.mark.parametrize("use_chirality", [True, False])
def test_comp_matching_chirality(use_chirality):
    smarts = ["C[C@H](N)C(=O)"]
    smiles = ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]
    params = Parameters(smarts, [use_chirality])
    ms = MatchingSubstructure(params)
    results = ms(smiles)
    if use_chirality:
        assert results.scores[0][0] == 1.0
        assert results.scores[0][1] == 0.5
    else:
        assert results.scores[0][0] == 1.0
        assert results.scores[0][1] == 1.0


@pytest.mark.parametrize("smarts", [["C(("], [["CO", "C(("]]])
def test_comp_matching_invalid_smarts(smarts):
    params = Parameters(smarts, [False])

    with pytest.raises(ValueError, match=r"C\(\("):
        MatchingSubstructure(params)
