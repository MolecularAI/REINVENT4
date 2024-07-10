import pytest

from reinvent.runmodes.scoring.run_scoring import run_scoring


@pytest.fixture
def setup(json_config):
    config = {
        "parameters": {"smiles_file": json_config["REINVENT_INCEPTION_SMI"]},
        "scoring": {
            "type": "custom_sum",
            "parallel": False,
            "component": [
                {
                    "custom_alerts": {
                        "endpoint": [
                            {
                                "name": "Unwanted SMARTS",
                                "weight": 0.79,
                                "params": {
                                    "smarts": [
                                        "[*;r8]",
                                        "[*;r9]",
                                        "[*;r10]",
                                        "[*;r11]",
                                        "[*;r12]",
                                        "[*;r13]",
                                        "[*;r14]",
                                        "[*;r15]",
                                        "[*;r16]",
                                        "[*;r17]",
                                        "[#8][#8]",
                                        "[#6;+]",
                                        "[#16][#16]",
                                        "[#7;!n][S;!$(S(=O)=O)]",
                                        "[#7;!n][#7;!n]",
                                        "C#C",
                                        "C(=[O,S])[O,S]",
                                        "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                        "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                                        "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                                        "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                        "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                                        "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                    ]
                                },
                            }
                        ]
                    }
                },
                {
                    "MolecularWeight": {
                        "endpoint": [
                            {
                                "name": "Molecular weight",
                                "weight": 0.342,
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
        },
    }

    return config


@pytest.mark.integration
def test_run_sampling_with_likelihood(setup):
    config = setup

    run_scoring(config)
