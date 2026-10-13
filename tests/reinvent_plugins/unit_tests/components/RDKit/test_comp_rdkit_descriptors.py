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


def test_comp_rdkit_descriptors_unknown_descriptor():
    params = Parameters(["unknown"])
    with pytest.raises(ValueError, match="unknown descriptor"):
        RDKitDescriptors(params)


def test_comp_rdkit_descriptors_none_molecule_scores_nan():
    # an invalid SMILES (None molecule from molcache) must score NaN rather
    # than raising (issue #333)
    params = Parameters(["MolWt"])
    component = RDKitDescriptors(params)

    results = component(["not_a_molecule", "c1ccccc1"])

    assert np.isnan(results.scores[0][0])
    assert results.scores[0][1] == pytest.approx(78.114)
