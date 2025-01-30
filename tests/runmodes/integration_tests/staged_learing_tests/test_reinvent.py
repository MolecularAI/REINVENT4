import os
import pytest
from pathlib import Path

import torch

from reinvent.runmodes.RL.run_staged_learning import run_staged_learning
from reinvent.runmodes.utils.helpers import set_torch_device


@pytest.fixture
def setup(tmp_path, json_config, pytestconfig):
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    config = {
        "parameters": {
            "use_checkpoint": False,
            "prior_file": ".reinvent",
            "agent_file": ".reinvent",
            "batch_size": 50,
            "unique_sequences": True,
            "randomize_smiles": True,
        },
        "learning_strategy": {"type": "dap", "sigma": 120, "rate": 0.0001},
        "diversity_filter": {
            "type": "IdenticalMurckoScaffold",
            "bucket_size": 25,
            "minscore": 0.4,
            "minsimilarity": 0.4,
            "penalty_multiplier": 0.5,
        },
        "stage": [
            {
                "chkpt_file": "test1.chkpt",
                "termination": "simple",
                "max_score": 0.7,
                "min_steps": 1,
                "max_steps": 5,
                "scoring": {
                    "type": "custom_product",
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
            },
        ],
    }
    return config


@pytest.mark.integration
def test_staged_learning(setup, tmp_path, pytestconfig):
    config = setup
    device = pytestconfig.getoption("device")

    device = torch.device(device)
    run_staged_learning(config, device, tb_logdir=None, responder_config=None)

    checkpoint_file = Path("test1.chkpt")

    assert checkpoint_file.exists()

    model = torch.load(checkpoint_file, weights_only=False)
    keys = list(model.keys())

    assert keys == [
        "max_sequence_length",
        "metadata",
        "model_type",
        "network",
        "network_params",
        "staged_learning",
        "tokenizer",
        "version",
        "vocabulary",
    ]
