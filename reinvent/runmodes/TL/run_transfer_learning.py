"""Reinvent transfer learning

Reads in a SMILES file and performs transfer learning.
"""

import os
import logging

import torch
import torch.optim as topt

from reinvent.runmodes import TL, create_adapter
from reinvent.utils import setup_reporter, read_smiles_csv_file
from reinvent.chemistry import conversions
from reinvent.chemistry.standardization.rdkit_standardizer import (
    RDKitStandardizer,
)
from .validation import TLConfig

logger = logging.getLogger(__name__)


def run_transfer_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str = None,
    write_config: str = None,
    *args,
    **kwargs,
):
    """Run transfer learning with Reinvent

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: log directory for TensorBoard
    :param write_config: callable to write config
    """

    logger.info("Starting Transfer Learning")

    config = TLConfig(**input_config)

    parameters = config.parameters
    scheduler_config = config.scheduler

    model_filename = parameters.input_model_file
    adapter, _, model_type = create_adapter(model_filename, "training", device)

    logger.info(f"Using generator {model_type}")

    smiles_filename = os.path.abspath(parameters.smiles_file)
    do_standardize = parameters.standardize_smiles

    randomize_all_smiles = parameters.randomize_all_smiles
    do_randomize = parameters.randomize_smiles and not randomize_all_smiles

    actions = []
    cols = 0

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
    smilies = read_smiles_csv_file(smiles_filename, cols, actions=actions, remove_duplicates=True)
    logger.info(f"Read {len(smilies)} input SMILES from {smiles_filename}")

    if not smilies:
        msg = f"Unable to read valid SMILES from {smiles_filename}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    validation_smiles_filename = parameters.validation_smiles_file
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

    if model_type == "Mol2Mol":
        model_size = adapter.network.encoder.layers[0].self_attn.linears[0].in_features
        lr_config = TL.LambdaLRConfiguration(**scheduler_config)

        optimizer = topt.Adam(
            adapter.get_network_parameters(),
            lr=lr_config.lr,
            betas=(lr_config.beta1, lr_config.beta2),
            eps=lr_config.eps,
            capturable=str(device) != "cpu",  # workaround for pytorch 1.11
        )

        lr_step = (
            lambda step: lr_config.factor
            / 1e-4
            * (
                model_size ** (-0.5)
                * min(
                    max(step, 1) ** (-0.5),
                    max(step, 1) * lr_config.warmup ** (-1.5),
                )
            )
        )

        lr_scheduler = topt.lr_scheduler.LambdaLR(optimizer, lr_step)
    else:
        lr_config = TL.StepLRConfiguration(**scheduler_config)
        optimizer = topt.Adam(adapter.get_network_parameters(), lr=lr_config.lr)

        lr_scheduler = topt.lr_scheduler.StepLR(
            optimizer, step_size=lr_config.step, gamma=lr_config.gamma
        )

    runner_class = getattr(TL, f"{model_type}")

    optimize = runner_class(
        adapter,
        smilies,
        validation_smilies,
        tb_logdir,
        parameters,
        optimizer,
        lr_scheduler,
        lr_config,
    )

    if "responder" in config:
        url = config.responder.endpoint
        success = setup_reporter(url)

        if success:
            logger.info(f"Remote reporting to {url}")

    if callable(write_config):
        write_config(config.model_dump())

    optimize()
