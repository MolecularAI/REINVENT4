import numpy as np
import pytest
from reinvent.scoring.scorer import Scorer
from numpy.testing import assert_array_almost_equal


@pytest.mark.integration
@pytest.mark.parametrize('use_pumas', [True, False])
def test_geo_scorer(use_pumas):
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_geo_mean = {
        "type": "geometric_mean",
        "use_pumas" : use_pumas,
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

    expected_result_geo_mean = [0.414504, 0.810673, 0.0, 0.0]
    geo_scorer = Scorer(scorer_config_geo_mean)
    geo_results = geo_scorer.compute_results(smiles, invalid_mask, duplicate_mask)

    assert_array_almost_equal(geo_results.total_scores, expected_result_geo_mean)
    assert (
        len(geo_results.completed_components) == 4
    )  # molecularweight, qed, custom alerts, matching subs

@pytest.mark.parametrize('use_pumas', [True, False])
def test_arth_scorer(use_pumas):
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_arth_mean = {
        "type": "arithmetic_mean",
        "use_pumas" : use_pumas,
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

    expected_result_arth_mean = [0.428014, 0.819214, 0.0, 0.0]
    arth_scorer = Scorer(scorer_config_arth_mean)
    arth_results = arth_scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    assert_array_almost_equal(arth_results.total_scores, expected_result_arth_mean)
    assert (
        len(arth_results.completed_components) == 4
    )  # molecularweight, qed, custom alerts, matching subs

@pytest.mark.parametrize('use_pumas', [True, False])
def test_filter_and_penalty(use_pumas):
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_filter_and_penalty = {
        "type": "geometric_mean",
        "use_pumas" : use_pumas,
        "component": [
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted SMARTS", "weight": 1, "params": {"smarts": ["F"]}}
                    ]
                }
            },
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

    expected_result_filter_and_penalty = [0.5, 1.0, 0.0, 0.0]
    filter_and_penalty_scorer = Scorer(scorer_config_filter_and_penalty)
    filter_and_penalty_results = filter_and_penalty_scorer.compute_results(
        smiles, invalid_mask, duplicate_mask
    )

    assert_array_almost_equal(
        filter_and_penalty_results.total_scores, expected_result_filter_and_penalty
    )
    assert (
        len(filter_and_penalty_results.completed_components) == 2
    )  # molecularweight, qed, custom alerts, matching subs

@pytest.mark.parametrize('use_pumas', [True, False])
def test_filter_scorer(use_pumas):
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_arth_mean = {
        "type": "arithmetic_mean",
        "use_pumas" : use_pumas,
        "component": [
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted SMARTS", "weight": 1, "params": {"smarts": ["F"]}}
                    ]
                }
            },
            {
                "NumAtomStereoCenters": {
                    "endpoint": [
                        {
                            "name": "Number of stereo centers",
                            "weight": 1,
                            "transform": {"type": "LeftStep", "low": 0},
                        }
                    ]
                }
            },
        ],
    }

    expected_result_arth_mean = [1.0, 1.0, 0.0, 0.0]
    arth_scorer = Scorer(scorer_config_arth_mean)
    arth_results = arth_scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    assert_array_almost_equal(arth_results.total_scores, expected_result_arth_mean)

@pytest.mark.parametrize('use_pumas', [True, False])
def test_all_filter_scorer(use_pumas):
    smiles = ["NCc1ccccc1", "NCc1ccccc1C(=O)O", "NCc1ccccc1C(F)", "NCc1ccccc1C(=O)F"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_arth_mean = {
        "type": "arithmetic_mean",
        "use_pumas" : use_pumas,
        "component": [
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted SMARTS", "weight": 1, "params": {"smarts": ["[#6]"]}}
                    ]
                }
            },
            {
                "NumAtomStereoCenters": {
                    "endpoint": [
                        {
                            "name": "Number of stereo centers",
                            "weight": 1,
                            "transform": {"type": "LeftStep", "low": 0},
                        }
                    ]
                }
            },
        ],
    }
    expected_result_arth_mean = [0.0, 0.0, 0.0, 0.0]
    arth_scorer = Scorer(scorer_config_arth_mean)
    arth_results = arth_scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    assert_array_almost_equal(arth_results.total_scores, expected_result_arth_mean)


@pytest.mark.integration
@pytest.mark.parametrize('use_pumas', [True, False])
def test_fragment_scoring(use_pumas):
    smilies = ["CC(C)(C(=O)O)n1cnc(NC=O)c1", "O=COc1ccc2cc(-c3ccc(C(=O)O)nn3)ccc2[n+]1CC(=O)O"]
    fragments = [
        "[*]Nc1cn(C(C)(C)C(=O)[*])cn1",
        "[*]Oc1ccc2cc(-c3ccc(C(=O)O)nn3)ccc2[n+]1CC(=O)[*]",
    ]
    invalid_mask = np.array([True, True])
    duplicate_mask = np.array([True, True])

    # test a full and fragment component
    scorer = Scorer(
        input_config={
            "type": "geometric_mean",
            "use_pumas" : use_pumas,
            "component": [
                {
                    "MolecularWeight": {
                        "endpoint": [
                            {
                                "name": "Molecular weight",
                                "weight": 1,
                                "transform": {
                                    "type": "double_sigmoid",
                                    "high": 500.0,
                                    "low": 200.0,
                                    "coef_div": 500.0,
                                    "coef_si": 20.0,
                                    "coef_se": 20.0,
                                },
                            }
                        ]
                    }
                },
                {
                    "FragmentMolecularWeight": {
                        "endpoint": [
                            {
                                "name": "Molecular weight",
                                "weight": 1,
                                "transform": {
                                    "type": "double_sigmoid",
                                    "high": 500.0,
                                    "low": 200.0,
                                    "coef_div": 500.0,
                                    "coef_si": 20.0,
                                    "coef_se": 20.0,
                                },
                            }
                        ]
                    }
                },
            ],
        }
    )

    res = scorer(
        smilies=smilies,
        invalid_mask=invalid_mask,
        duplicate_mask=duplicate_mask,
        fragments=fragments,
    )
    assert np.allclose(
        res.completed_components[0].component_result.fetch_scores(smilies, transpose=True),
        [197.194, 354.2980000000001],
    )
    assert np.allclose(
        res.completed_components[1].component_result.fetch_scores(smilies, transpose=True),
        [153.185, 310.28900000000004],
    )


@pytest.mark.integration
@pytest.mark.parametrize('use_pumas', [True, False])
def test_libinvent_scoring(use_pumas):
    smilies = ["CC(Oc1ccc(C(C)C)cc1)C(=O)NCCCCc1ccc(N(C)C)cc1", "O=CNCCCCc1ccc(-c2ccc(O)cc2)cc1"]
    connectivity_annotated_smiles = [
        "CC(Oc1ccc(C(C)C)cc1)[C:0](=O)[NH:0]CCCCc1cc[c:1]([N:1](C)C)cc1",
        "O=[CH:0][NH:0]CCCCc1cc[c:1](-[c:1]2ccc(O)cc2)cc1",
    ]
    invalid_mask = np.array([True, True])
    duplicate_mask = np.array([True, True])

    # test a full and fragment component
    scorer = Scorer(
        input_config={
            "type": "geometric_mean",
            "use_pumas" : use_pumas,
            "component": [
                {
                    "MolecularWeight": {
                        "endpoint": [
                            {
                                "name": "Molecular weight",
                                "weight": 1,
                                "transform": {
                                    "type": "double_sigmoid",
                                    "high": 500.0,
                                    "low": 200.0,
                                    "coef_div": 500.0,
                                    "coef_si": 20.0,
                                    "coef_se": 20.0,
                                },
                            }
                        ]
                    }
                },
                {
                    "ReactionFilter": {
                        "endpoint": [
                            {
                                "name": "amide coupling/suzuki",
                                "params": {
                                    "type": "selective",
                                    "reaction_smarts": [
                                        [
                                            "[C:2]([#7;!D4:1])(=[O:3])[#6:4]>>[#7:1][*].[C,$(C=O):2](=[O:3])([*])[#6:4]"
                                        ],
                                        [
                                            "[c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1]-!@[N;$(NC)&!$(N=*)&!$([N-])&!$(N#*)&!$([ND1])&!$(N[O])&!$(N[C,S]=[S,O,N]),H2&$(Nc1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):2]>>[*][c;$(c1:[c,n]:[c,n]:[c,n]:[c,n]:[c,n]:1):1].[*][N:2]"
                                        ],
                                    ]},
                                "weight": 1,
                            }
                        ]
                    }
                },
            ],
        }
    )

    res = scorer(
        smilies=smilies,
        invalid_mask=invalid_mask,
        duplicate_mask=duplicate_mask,
        connectivity_annotated_smiles = connectivity_annotated_smiles,
    )
    assert np.allclose(
        res.completed_components[0].component_result.fetch_scores(smilies, transpose=True),
        [382.5480000000001, 0.0],
    )

@pytest.mark.integration
@pytest.mark.parametrize('use_pumas', [True, False])
def test_metadata_passing(use_pumas):
    smiles = ["FCc1ccccc1", "SCc1ccccc1", "OCc1ccccc1", "O=Cc1ccccc1"]
    invalid_mask = np.array([True, True, True, True])
    duplicate_mask = np.array([True, True, True, True])

    scorer_config_geo_mean = {
        "type": "geometric_mean",
        "use_pumas" : use_pumas,
        "component": [
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted F", "weight": 1, "params": {"smarts": ["F"]}}
                    ]
                }
            },
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted S", "weight": 1, "params": {"smarts": ["S"]}}
                    ]
                }
            },
            {
                "custom_alerts": {
                    "endpoint": [
                        {"name": "Unwanted CO", "weight": 1, "params": {"smarts": ["CO"]}}
                    ]
                }
            }
        ],
    }

    metadata_res_values = [["['F']", '[]', '[]', '[]'], ['None', "['S']", '[]', '[]'], ['None', 'None', "['CO']", '[]']]
    metadata_res_names = ['matchting_patterns (Unwanted F)', 'matchting_patterns (Unwanted S)',
                          'matchting_patterns (Unwanted CO)']

    expected_result_geo_mean = [0, 0, 0, 1]

    geo_scorer = Scorer(scorer_config_geo_mean)
    geo_results = geo_scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    metadata_values = []
    metadata_names = []
    for transformed_result in geo_results.completed_components:
        for _metadata_name, _metadata_value in transformed_result.component_result.fetch_metadata(
                geo_results.smilies
        ).items():
            metadata_values.append([str(val) for val in _metadata_value])
            metadata_names.append(f"{_metadata_name} ({transformed_result.component_names[0]})")
    assert metadata_values == metadata_res_values
    assert metadata_names == metadata_res_names
    assert_array_almost_equal(geo_results.total_scores, expected_result_geo_mean)

