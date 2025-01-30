import os
import shutil
import pytest
import unittest

import torch

from reinvent.runmodes.create_model.create_libinvent import create_model


@pytest.mark.integration
@pytest.mark.usefixtures("json_config")
class TestCreateModel(unittest.TestCase):
    def setUp(self):
        self.workdir = self.json_config["MAIN_TEST_PATH"]
        self.output_file = os.path.join(self.workdir, "libinvent_empty.model")

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
            input_smiles_path=self.json_config["LIBINVENT_SMILES_SET_PATH"],
            output_model_path=self.output_file,
        )

        model = torch.load(self.output_file, weights_only=False)
        keys = list(model.keys())

        self.assertEqual(
            keys,
            ["model_type", "version", "metadata", "model", "decorator"],
        )

        self.assertEqual(model["model_type"], "Libinvent")
        self.assertEqual(model["model"]["max_sequence_length"], 256)

        encoder_params = model["decorator"]["params"]["encoder_params"]
        self.assertEqual(encoder_params["num_dimensions"], 512)
        self.assertEqual(encoder_params["num_layers"], 3)
        self.assertEqual(encoder_params["vocabulary_size"], 30)  # data dependant
        self.assertEqual(encoder_params["dropout"], 0.0)

        decoder_params = model["decorator"]["params"]["decoder_params"]
        self.assertEqual(decoder_params["num_dimensions"], 512)
        self.assertEqual(decoder_params["num_layers"], 3)
        self.assertEqual(decoder_params["vocabulary_size"], 29)  # data dependant
        self.assertEqual(decoder_params["dropout"], 0.0)
