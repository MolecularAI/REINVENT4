"""Reinvent transfer learning

Reads in a SMILES file and performs transfer learning.
"""

import os
import logging

import torch
import torch.optim as topt

from reinvent.runmodes import TL, create_adapter
from reinvent.config_parse import read_smiles_csv_file
from reinvent.runmodes.reporter.remote import setup_reporter, get_reporter
from reinvent.chemistry import Conversions
from reinvent.chemistry.standardization.rdkit_standardizer import (
    RDKitStandardizer,
)

logger = logging.getLogger(__name__)


def run_transfer_learning(
    config: dict, device: torch.device, tb_logdir: str = None, *args, **kwargs
):
    """Run transfer learning with Reinvent

    :param config: the run configuration
    :param device: torch device
    :param tb_logdir: log directory for TensorBoard
    """

    logger.info("Starting Transfer Learning")

    parameters = config["parameters"]

    model_filename = parameters["input_model_file"]
    adapter, _, model_type = create_adapter(model_filename, "training", device)

    logger.info(f"Using generator {model_type}")

    smiles_filename = os.path.abspath(parameters["smiles_file"])
    do_standardize = parameters.get("standardize_smiles", True)

    randomize_all_smiles = parameters.get("randomize_all_smiles", False)
    do_randomize = parameters.get("randomize_smiles", True) and not randomize_all_smiles

    internal_diversity = parameters.get("internal_diversity", False)

    actions = []
    cols = 0
    conversions = Conversions()

    # FIXME: move to preprocessing
    if model_type == "Reinvent":
        if do_standardize:
            standardizer = RDKitStandardizer(None, isomeric=False)
            actions.append(standardizer.apply_filter)

        if do_randomize:
            actions.append(conversions.randomize_smiles)
    elif model_type == "Mol2Mol":
        if do_standardize:
            actions.append(conversions.convert_to_standardized_smiles)
    else:
        cols = slice(0, 2, None)

    # NOTE: we expect here that all data will fit into memory
    smilies = read_smiles_csv_file(
        smiles_filename, cols, actions=actions, remove_duplicates=True
    )
    logger.info(f"Read {len(smilies)} input SMILES from {smiles_filename}")

    if not smilies:
        msg = f"Unable to read valid SMILES from {smiles_filename}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    validation_smiles_filename = parameters.get("validation_smiles_file")
    validation_smilies = None

    if validation_smiles_filename:
        validation_smiles_filename = os.path.abspath(validation_smiles_filename)
        validation_smilies = read_smiles_csv_file(
            validation_smiles_filename,
            cols,
            actions=actions,
            remove_duplicates=True,
        )
        logger.info(
            f"Read {len(validation_smilies)} validation SMILES from {validation_smiles_filename}"
        )

    common_opts = dict(
        input_model_file=model_filename,
        output_model_file=parameters.get("output_model_file", "TL.model"),
        smilies=smilies,
        validation_smilies=validation_smilies,
        batch_size=parameters["batch_size"],
        num_epochs=parameters["num_epochs"],
        save_every_n_epochs=parameters["save_every_n_epochs"],
        sample_batch_size=parameters.get("sample_batch_size", 1),
        num_refs=parameters.get("num_refs", 0),
        n_cpus=config.get("number_of_cpus", 1),
        randomize_all_smiles=randomize_all_smiles,
        internal_diversity=internal_diversity,
    )

    if model_type == "Mol2Mol":
        model_size = adapter.network.encoder.layers[0].self_attn.linears[0].in_features
        noam_config = TL.NoamoptConfiguration()
        adam = topt.Adam(
            adapter.get_network_parameters(),
            lr=noam_config.lr,
            betas=(noam_config.beta1, noam_config.beta2),
            eps=noam_config.eps,
            capturable=str(device) != "cpu",  # workaround for pytorch 1.11
        )

        lr_step = (
            lambda step: noam_config.factor
            / 1e-4
            * (
                model_size ** (-0.5)
                * min(
                    max(step, 1) ** (-0.5),
                    max(step, 1) * noam_config.warmup ** (-1.5),
                )
            )
        )
        learning_rate_scheduler = topt.lr_scheduler.LambdaLR(adam, lr_step)

        mode_config = TL.Mol2MolConfiguration(
            **common_opts,
            optimizer=adam,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_config=TL.StepLRConfiguration(min=1e-10),
            pairs=parameters["pairs"],
            max_sequence_length=parameters.get("max_sequence_length", None),
            ranking_loss_penalty=parameters.get("ranking_loss_penalty", False),
        )
    else:
        lr_config = TL.StepLRConfiguration(step=10, min=0.0000001)
        adam = topt.Adam(adapter.get_network_parameters(), lr=lr_config.start)
        learning_rate_scheduler = topt.lr_scheduler.StepLR(
            adam, step_size=lr_config.step, gamma=lr_config.gamma
        )

        mode_config = TL.GeneralConfiguration(
            **common_opts,
            optimizer=adam,
            learning_rate_scheduler=learning_rate_scheduler,
            learning_rate_config=lr_config,
        )

    runner_class = getattr(TL, f"{model_type}")
    optimize = runner_class(adapter, tb_logdir, mode_config)

    if "logging" in config:
        url = config["logging"].get("endpoint", None)
        setup_reporter(url)
        reporter = get_reporter()

        if hasattr(reporter, "url"):
            logger.info(f"Remote reporting to {reporter.url}")

    optimize()
