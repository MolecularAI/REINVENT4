import pytest
import numpy as np

from reinvent_plugins.components.RDKit.comp_rdkit_descriptors import Parameters, RDKitDescriptors


def test_comp_rdkit_descriptors():
    input_smiles = ["c1ccccc1", "CCC#CCC"]
    component_result = {
        "qed": np.array([0.44262837, 0.39083325]),
        "MolWt": np.array([78.114, 82.146]),
        "NumHAcceptors": np.array([0.0, 0.0]),
        "NumHDonors": np.array([0.0, 0.0]),
        "NumRotatableBonds": np.array([0.0, 0.0]),
        "FractionCSP3": np.array([0.0, 2 / 3.0]),
        "HeavyAtomCount": np.array([6.0, 6.0]),
        "NumHeteroatoms": np.array([0.0, 0.0]),
        "RingCount": np.array([1.0, 0.0]),
        "NumAromaticRings": np.array([1.0, 0.0]),
        "NumAliphaticRings": np.array([0.0, 0.0]),
        "MolLogP": np.array([1.6866, 1.8098]),
        "fr_amide": np.array([0.0, 0.0]),
        "BertzCT": np.array([71.96100506, 53.8428331]),
    }

    for component, result in component_result.items():
        params = Parameters([component])
        component = RDKitDescriptors(params)

        results = component(input_smiles)

        assert np.allclose(results.scores[0], result)


@pytest.mark.xfail
def test_comp_rdkit_descriptors_unknown_descriptor():
    params = Parameters(["unknown"])
