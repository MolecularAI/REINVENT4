import numpy as np
import pytest
from reinvent.scoring.scorer import Scorer
from numpy.testing import assert_array_almost_equal


@pytest.mark.integration
def test_geo_scorer():
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_geo_mean = {
        "type": "geometric_mean",
        "component": [
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted SMARTS", "weight": 1, "params": {"smarts": ["F"]}}
                    ]
                }
            },
            {
                "MolecularWeight": {
                    "endpoint": [
                        {
                            "name": "Molecular weight",
                            "weight": 1,
                            "transform": {
                                "type": "double_sigmoid",
                                "high": 175.0,
                                "low": 25.0,
                                "coef_div": 500.0,
                                "coef_si": 20.0,
                                "coef_se": 20.0,
                            },
                        }
                    ]
                }
            },
            {"QED": {"endpoint": [{"name": "QED", "weight": 0.5}]}},
            {
                "MatchingSubstructure": {
                    "endpoint": [
                        {
                            "name": "MatchingSubstructure inline C=O",
                            "weight": 1,
                            "params": {"smarts": "C=O", "use_chirality": False},
                        }
                    ]
                }
            },
        ],
    }

    expected_result_geo_mean = [0.414361, 0.810668, 0.425256, 0.796556]
    geo_scorer = Scorer(scorer_config_geo_mean)
    geo_results = geo_scorer.compute_results(smiles, invalid_mask, duplicate_mask)

    assert_array_almost_equal(geo_results.total_scores, expected_result_geo_mean)
    assert (
        len(geo_results.completed_components) == 4
    )  # molecularweight, qed, custom alerts, matching subs


def test_arth_scorer():
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_arth_mean = {
        "type": "arithmetic_mean",
        "component": [
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted SMARTS", "weight": 1, "params": {"smarts": ["F"]}}
                    ]
                }
            },
            {
                "MolecularWeight": {
                    "endpoint": [
                        {
                            "name": "Molecular weight",
                            "weight": 1,
                            "transform": {
                                "type": "double_sigmoid",
                                "high": 175.0,
                                "low": 25.0,
                                "coef_div": 500.0,
                                "coef_si": 20.0,
                                "coef_se": 20.0,
                            },
                        }
                    ]
                }
            },
            {"QED": {"endpoint": [{"name": "QED", "weight": 0.5}]}},
            {
                "MatchingSubstructure": {
                    "endpoint": [
                        {
                            "name": "MatchingSubstructure inline C=O",
                            "weight": 1,
                            "params": {"smarts": "C=O", "use_chirality": False},
                        }
                    ]
                }
            },
        ],
    }

    expected_result_arth_mean = [0.427842, 0.819208, 0.431715, 0.804572]
    arth_scorer = Scorer(scorer_config_arth_mean)
    arth_results = arth_scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    assert_array_almost_equal(arth_results.total_scores, expected_result_arth_mean)
    assert (
        len(arth_results.completed_components) == 4
    )  # molecularweight, qed, custom alerts, matching subs
