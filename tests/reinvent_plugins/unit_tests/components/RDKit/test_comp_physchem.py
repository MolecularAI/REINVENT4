import numpy as np

from reinvent_plugins.components.RDKit.comp_physchem import Qed
from reinvent_plugins.components.RDKit.comp_physchem import MolecularWeight
from reinvent_plugins.components.RDKit.comp_physchem import GraphLength
from reinvent_plugins.components.RDKit.comp_physchem import NumAtomStereoCenters
from reinvent_plugins.components.RDKit.comp_physchem import HBondAcceptors
from reinvent_plugins.components.RDKit.comp_physchem import HBondDonors
from reinvent_plugins.components.RDKit.comp_physchem import NumRotBond
from reinvent_plugins.components.RDKit.comp_physchem import Csp3
from reinvent_plugins.components.RDKit.comp_physchem import numsp, numsp2, numsp3
from reinvent_plugins.components.RDKit.comp_physchem import NumHeavyAtoms
from reinvent_plugins.components.RDKit.comp_physchem import NumHeteroAtoms
from reinvent_plugins.components.RDKit.comp_physchem import NumRings
from reinvent_plugins.components.RDKit.comp_physchem import NumAromaticRings
from reinvent_plugins.components.RDKit.comp_physchem import NumAliphaticRings
from reinvent_plugins.components.RDKit.comp_physchem import SlogP


def test_comp_physchem():
    input_smiles = ["c1ccccc1", "CCC#CCC"]
    expected_results = {
        Qed: np.array([0.44262837, 0.39083325]),
        MolecularWeight: np.array([78.114, 82.146]),
        GraphLength: np.array([3.0, 5.0]),
        NumAtomStereoCenters: np.array([0.0, 0.0]),
        HBondAcceptors: np.array([0.0, 0.0]),
        HBondDonors: np.array([0.0, 0.0]),
        NumRotBond: np.array([0.0, 0.0]),
        Csp3: np.array([0.0, 2 / 3.0]),
        numsp: np.array([0.0, 2.0]),
        numsp2: np.array([6.0, 0.0]),
        numsp3: np.array([0.0, 4.0]),
        NumHeavyAtoms: np.array([6.0, 6.0]),
        NumHeteroAtoms: np.array([0.0, 0.0]),
        NumRings: np.array([1.0, 0.0]),
        NumAromaticRings: np.array([1.0, 0.0]),
        NumAliphaticRings: np.array([0.0, 0.0]),
        SlogP: np.array([1.6866, 1.8098]),
    }

    for component in expected_results:
        results = component()(input_smiles)
        assert np.allclose(results.scores[0], expected_results[component])
