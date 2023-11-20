import pytest

import numpy as np

from reinvent_plugins.components.Lilly.comp_medchem_rules import Parameters
from reinvent_plugins.components.Lilly.comp_medchem_rules import LillyMedchemRules

SMILIES = [
    "Cc1ccc(-n2[nH]c(C)c(N=Nc3cccc(-c4cccc(C(=O)O)c4)c3O)c2=O)cc1C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",
    "Cc1c(N(C)CS(=O)(=O)O)c(=O)n(-c2ccccc2)n1C",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C",
    "Clc1ccc2c(c1)C(N1CCNCC1)=Nc1ccccc1O2",
    "CNC(C)C1CCC(N)C(OC2C(N)CC(N)C(OC3OCC(C)(O)C(NC)C3O)C2O)O1",
]


@pytest.mark.parametrize(
    "relaxed, expected_results",
    [
        (([False],),
         [153, 999, 6, 0, 40, 0, 70, 50, 253]),
        (([True],),
         [189, 999, 0, 0, 40, 0, 70, 50, 229]),
    ],
)
def test_comp_lilly_medchem_rules(relaxed, expected_results):
    params = Parameters(*relaxed)
    rules = LillyMedchemRules(params)
    results = rules(SMILIES)
    expected = np.array(expected_results)

    assert (results.scores == np.array(expected_results)).all()
