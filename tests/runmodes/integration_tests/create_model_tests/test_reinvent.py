import os
import shutil
import pytest
import unittest

import torch

from reinvent.runmodes.create_model.create_reinvent import create_model


@pytest.mark.integration
@pytest.mark.usefixtures("json_config")
class TestCreateModel(unittest.TestCase):
    def setUp(self):
        self.workdir = self.json_config["MAIN_TEST_PATH"]
        self.output_file = os.path.join(self.workdir, "reinvent_empty.model")

        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir)

    def tearDown(self):
        if os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir)

    def test_create_model(self):
        create_model(
            num_layers=3,
            layer_size=512,
            dropout=0.0,
            max_sequence_length=256,
            cell_type="lstm",
            embedding_layer_size=256,
            layer_normalization=False,
            standardize=True,
            input_smiles_path=self.json_config["SMILES_SET_PATH"],
            output_model_path=self.output_file,
            metadata={"data_source": "pytest", "comment": "pytest"},
        )

        model = torch.load(self.output_file, weights_only=False)
        keys = list(model.keys())

        self.assertEqual(
            keys,
            [
                "model_type",
                "version",
                "metadata",
                "vocabulary",
                "tokenizer",
                "max_sequence_length",
                "network",
                "network_params",
            ],
        )

        self.assertEqual(model["model_type"], "Reinvent")
        self.assertEqual(model["max_sequence_length"], 256)

        network_params = model["network_params"]
        self.assertEqual(network_params["dropout"], 0.0)
        self.assertEqual(network_params["layer_size"], 512)
        self.assertEqual(network_params["num_layers"], 3)
        self.assertEqual(network_params["cell_type"], "lstm")
        self.assertEqual(network_params["embedding_layer_size"], 256)
