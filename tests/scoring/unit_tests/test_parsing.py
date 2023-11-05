from reinvent.scoring.scorer import get_components
from dataclasses import fields

def test_get_components():
    components = [
        {"QED": {"endpoint": [{"name": "QED drug-like score", "weight": 0.79}]}},
         {"MolecularWeight": {
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
        }}
    ]

    components_dict = get_components(components)

    assert len(components_dict.scorers) == 2
    assert "qed" == components_dict.scorers[0].component_type
    assert "molecularweight" == components_dict.scorers[1].component_type
    for component in components_dict.scorers:
        assert len(fields(component)) == 3 # name, params, cache
        assert len(component.params) == 4 # name, component object, transform, weight

    assert not components_dict.filters
    assert not components_dict.penalties
