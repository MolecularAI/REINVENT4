import pytest

import numpy as np

from reinvent_plugins.components.comp_reaction_filter import Parameters
from reinvent_plugins.components.comp_reaction_filter import ReactionFilter


@pytest.mark.parametrize(
    "reaction_type, expected_results",
    [
        ("selective", [0.0, 0.0]),
        ("nonselective", [0.0, 0.0]),
    ],
)
def test_comp_reaction_filter(reaction_type, expected_results):
    reaction_smarts = "[C:2]([#7;!D4:1])(=[O:3])[#6:4]>>[#7:1][*].[C,$(C=O):2](=[O:3])([*])[#6:4]"
    params = Parameters([reaction_type], [[[reaction_smarts]]])
    rs = ReactionFilter(params)
    results = rs(["[*:0]C(=O)Cn1c2ccncc2c3cccnc13", "[*:0]Cc2ccc1cncc(C[*:1])c1c2"])

    assert (results.scores == np.array(expected_results)).all()


def test_comp_reaction_filter_amide_coupling():
    reaction_smarts = "[C:2]([#7;!D4:1])(=[O:3])[#6:4]>>[#7:1][*].[C,$(C=O):2](=[O:3])([*])[#6:4]"
    params = Parameters(["selective"], [[[reaction_smarts]]])
    rs = ReactionFilter(params)
    results = rs(["CC[C:0](=O)[N:0]CC", "CC[C:0](=O)[C:0]CC"])

    assert (results.scores == np.array([1.0, 0.0])).all()


def test_comp_reaction_filter_multiple_attachments():
    reaction_smarts = "[C:2]([#7;!D4:1])(=[O:3])[#6:4]>>[#7:1][*].[C,$(C=O):2](=[O:3])([*])[#6:4]"
    params = Parameters(["selective"], [[[reaction_smarts]]])
    with pytest.raises(ValueError):
        rs = ReactionFilter(params)
        results = rs(["CC[CH:0]([CH3:1])[N:0]CC"])
