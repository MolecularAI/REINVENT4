import os
import shutil
import pytest
import unittest

import torch

from reinvent.runmodes.create_model.create_mol2mol import create_model


@pytest.mark.integration
@pytest.mark.usefixtures("json_config")
class TestCreateModel(unittest.TestCase):
    def setUp(self):
        self.workdir = self.json_config["MAIN_TEST_PATH"]
        self.output_file = os.path.join(self.workdir, "molformer_empty.model")

        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir)

    def tearDown(self):
        if os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir)

    def test_create_model(self):
        create_model(
            num_layers=6,
            num_heads=8,
            model_dimension=256,
            feedforward_dimension=2048,
            dropout=0.0,
            max_sequence_length=256,
            input_smiles_path=self.json_config["MOLFORMER_SMILES_SET_PATH"],
            output_model_path=self.output_file,
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
                "max_sequence_length",
                "network_parameter",
                "network_state",
            ],
        )

        self.assertEqual(model["model_type"], "Mol2Mol")
        self.assertEqual(model["max_sequence_length"], 256)

        network_parameter = model["network_parameter"]
        self.assertEqual(network_parameter["feedforward_dimension"], 2048)
        self.assertEqual(network_parameter["model_dimension"], 256)
        self.assertEqual(network_parameter["num_layers"], 6)
        self.assertEqual(network_parameter["num_heads"], 8)
        # self.assertEqual(network_parameter["vocabulary_size"], 39)  # data dependant
        self.assertEqual(network_parameter["dropout"], 0.0)
