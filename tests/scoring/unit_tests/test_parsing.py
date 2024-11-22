from reinvent.scoring.config import collect_params
from reinvent.scoring.scorer import get_components
from dataclasses import fields


def test_get_components():
    components = [
        {"QED": {"endpoint": [{"name": "QED drug-like score", "weight": 0.79}]}},
        {
            "MolecularWeight": {
                "endpoint": [
                    {
                        "name": "MW",
                        "transform": {
                            "type": "reverse_sigmoid",
                            "high": 1.2,
                            "low": 0.85,
                            "k": 0.2,
                        },
                        "weight": 0.79,
                    }
                ]
            }
        },
    ]

    components_dict = get_components(components)

    assert len(components_dict.scorers) == 2
    assert "qed" == components_dict.scorers[0].component_type
    assert "molecularweight" == components_dict.scorers[1].component_type
    for component in components_dict.scorers:
        assert len(fields(component)) == 3  # name, params, cache
        assert len(component.params) == 4  # name, component object, transform, weight

    assert not components_dict.filters
    assert not components_dict.penalties


def test_complevel_params():
    components = [
        {"QED": {"endpoint": [{"name": "QED drug-like score", "weight": 0.79}]}},
        {
            "external_process": {
                "params": {  # Component-level params.
                    "args": "--loglevel DEBUG",
                },
                "endpoint": [
                    {
                        "name": "Endpoint1",
                        "weight": 0.5,
                        "params": {  # Endpoint-level params.
                            "executable": "path/to/executable1",
                        },
                    },
                    {
                        "name": "Endpoint2",
                        "weight": 0.7,
                        "params": {
                            "executable": "path/to/executable2",
                            "args": "--loglevel INFO",
                        },
                    },
                ],
            }
        },
    ]

    components_dict = get_components(components)

    assert len(components_dict.scorers) == 2
    assert "qed" == components_dict.scorers[0].component_type
    assert "externalprocess" == components_dict.scorers[1].component_type
    name, comp, transform, weights = components_dict.scorers[1].params
    assert comp.executables == ["path/to/executable1", "path/to/executable2"]
    assert comp.args == ["--loglevel DEBUG", "--loglevel INFO"]

    assert not components_dict.filters
    assert not components_dict.penalties


def test_collect_params():
    # Test case 1: multiple dictionaries
    params = [
        {"x": 1, "y": 1},
        {"y": 2, "z": 2},
        {"z": 3, "x": 3},
    ]
    expected_output = {
        "x": [1, None, 3],
        "y": [1, 2, None],
        "z": [None, 2, 3],
    }
    assert collect_params(params) == expected_output

    # Test case 2: Empty list
    params = []
    expected_output = {}
    assert collect_params(params) == expected_output

    # Test case 3: List with one dictionary
    params = [{"key1": "value1"}]
    expected_output = {"key1": ["value1"]}
    assert collect_params(params) == expected_output

    # Test case 4: List with dictionaries having different keys
    params = [
        {"key1": "value1"},
        {"key2": "value2"},
        {"key3": "value3"}
    ]
    expected_output = {
       "key1": ["value1", None, None],
        "key2": [None, "value2", None],
        "key3": [None, None, "value3"]
    }
    assert collect_params(params) == expected_output
