from pathlib import Path
import pytest

import torch

from reinvent.runmodes.TL.run_transfer_learning import run_transfer_learning
from reinvent.runmodes.utils.helpers import set_torch_device
from reinvent.models.meta_data import check_valid_hash


@pytest.fixture
def setup(tmp_path, json_config, pytestconfig):
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    output_model_file = tmp_path / "TL_libinvent.model"

    config = {
        "parameters": {
            "input_model_file": ".libinvent",
            "smiles_file": json_config["TL_LIBINVENT_SMILES_PATH"],
            "output_model_file": str(output_model_file),
            "save_every_n_epochs": 2,
            "batch_size": 64,
            "sample_batch_size": 100,
            "num_epochs": 2,
            "num_refs": 2,
        }
    }

    return config


@pytest.mark.integration
def test_transfer_learning(setup, tmp_path, pytestconfig):
    config = setup
    device = pytestconfig.getoption("device")

    run_transfer_learning(config, device)

    checkpoint_files = list(Path(tmp_path).glob("*.chkpt"))

    assert len(checkpoint_files) == 1

    model = torch.load(config["parameters"]["output_model_file"], weights_only=False)
    keys = list(model.keys())

    assert keys == ["decorator", "metadata", "model", "model_type", "version"]

    assert model["model_type"] == "Libinvent"
    assert check_valid_hash(model)
