import pytest

import numpy as np

from reinvent_plugins.components.RDKit_extra.comp_unwanted_substructures import Parameters
from reinvent_plugins.components.RDKit_extra.comp_unwanted_substructures import UnwantedSubstructures

SMILIES = [
    "CC1=C(C=C(C=C1)N2C(=O)C(=C(N2)C)N=NC3=CC=CC(=C3O)C4=CC(=CC=C4)C(=O)O)C", # Eltrombopag
    "O=C(C)Oc1ccccc1C(=O)O",  # Aspirin
    "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N",  # Celecoxib
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
    "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O",  # Metamizole
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC",  # Cocaine
    "C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl",  # Amoxapine
    "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC"  # Gentamicin
]

@pytest.mark.parametrize(
    "catalog_names, expected_results",
    [
        ([["PAINS"]], [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ([["BRENK", "NIH"]], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
        ([["ALL"]] , [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
)

def test_comp_unwanted_substructures(catalog_names, expected_results):
    params = Parameters(catalog_names)
    unwanted_substructures = UnwantedSubstructures(params)
    results = unwanted_substructures(SMILIES)

    assert (results.scores == np.array(expected_results)).all()
