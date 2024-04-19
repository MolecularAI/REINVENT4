"""Sample SMILES from a model

LibInvent, LinkInvent and Mol2Mol need input SMILES while Reinvent does not.
The output is a list of SMILES including the fragments where applicable and
the negative log likelihood if requested.
"""

__all__ = ["run_sampling"]
import logging
import os
import time


from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


from reinvent import setup_logger, CsvFormatter
from reinvent.runmodes import create_adapter
from reinvent.runmodes.samplers.reports.remote import setup_RemoteData, send_report
from reinvent.runmodes.samplers.reports.tensorboard import (
    setup_TBData,
    write_report as tb_write_report,
)
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.dtos import ChemistryHelpers
from reinvent.config_parse import read_smiles_csv_file
from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState
from reinvent.chemistry import Conversions
from reinvent.chemistry.library_design import BondMaker, AttachmentPoints

logger = logging.getLogger(__name__)

HEADERS = {
    "Reinvent": ("SMILES", "NLL"),
    "Libinvent": ("SMILES", "Scaffold", "R-groups", "NLL"),
    "Linkinvent": ("SMILES", "Warheads", "Linker", "NLL"),
    "LinkinventTransformer": ("SMILES", "Warheads", "Linker", "NLL"),
    "Mol2Mol": ("SMILES", "Input_SMILES", "Tanimoto", "NLL"),
}

FRAGMENT_GENERATORS = ["Libinvent", "Linkinvent", "LinkinventTransformer"]


def run_sampling(config: dict, device, *args, **kwargs):
    """Sampling run setup"""

    logger.info("Starting Sampling")

    parameters = config["parameters"]
    smiles_output_filename = parameters["output_file"]

    agent_model_filename = parameters["model_file"]
    adapter, _, model_type = create_adapter(agent_model_filename, "inference", device)

    logger.info(f"Using generator {model_type}")
    logger.info(f"Writing sampled SMILES to CSV file {smiles_output_filename}")

    csv_logger = setup_logger(
        name="csv",
        filename=smiles_output_filename,
        formatter=CsvFormatter(),
        propagate=False,
        level="INFO",
    )

    chemistry = ChemistryHelpers(
        Conversions(),  # Lib/LinkInvent, Mol2Mol
        BondMaker(),  # LibInvent
        AttachmentPoints(),  # Lib/LinkInvent
    )

    # number of smiles to be generated for each input; consistent with batch_size parameter as used in RL
    # different from batch size used in dataloader which affect cuda memory
    parameters["batch_size"] = parameters["num_smiles"]
    sampler, batch_size = setup_sampler(model_type, parameters, adapter, chemistry)
    sampler.unique_sequences = False

    try:
        smiles_input_filename = parameters["smiles_file"]
    except KeyError:
        smiles_input_filename = None

    input_smilies = None
    num_input_smilies = 1

    if smiles_input_filename:
        input_smilies = read_smiles_csv_file(smiles_input_filename, columns=0)
        num_input_smilies = len(input_smilies)

    num_total_smilies = parameters["num_smiles"] * num_input_smilies

    logger.info(f"Sampling {num_total_smilies} SMILES from model {agent_model_filename}")

    # NOTE: for beamsearch the batch size determines the beam size
    if model_type == "Mol2Mol" and parameters["sample_strategy"] == "beamsearch":
        if parameters["num_smiles"] > 300:
            logger.warning(f"Sampling with beam search may be very slow")

    # Time took fro sampling
    start_time = time.time()
    with torch.no_grad():
        sampled = sampler.sample(input_smilies)
    seconds_took = int(time.time() - start_time)
    logger.info(f"Time taken in seconds: {seconds_took}")

    kwargs = {}
    if model_type == "Mol2Mol":
        # compute Tanimoto similarity between generated compounds and input compounds; return largest
        valid_mols, valid_idxs = chemistry.conversions.smiles_to_mols_and_indices(sampled.items2)
        valid_scores = sampler.calculate_tanimoto(input_smilies, sampled.items2)
        scores = [None] * len(sampled.items2)
        for i, j in enumerate(valid_idxs):
            scores[j] = valid_scores[i]
        kwargs = {"Tanimoto": scores}

    # Log to tensorboard
    tb_logdir = parameters.get("tb_logdir", None)
    if tb_logdir:
        tb_reporter = SummaryWriter(log_dir=tb_logdir)
        tb_data = setup_TBData(sampled, seconds_took, **kwargs)
        tb_write_report(tb_reporter, tb_data)

    # Log to remote
    remote_data = setup_RemoteData(sampled, seconds_took, **kwargs)
    send_report(remote_data)

    # Unique canonical smiles
    unique_molecules = parameters.get("unique_molecules", True)
    if unique_molecules:
        sampled = filter_valid(sampled)

    # Write to csv
    csv_logger.info(HEADERS[model_type])
    if model_type == "Reinvent":
        records = zip(sampled.smilies, sampled.nlls.cpu().tolist())
    elif model_type in FRAGMENT_GENERATORS:
        records = zip(sampled.smilies, sampled.items1, sampled.items2, sampled.nlls.cpu().tolist())
    elif model_type == "Mol2Mol":
        records = zip(sampled.smilies, sampled.items1, scores, sampled.nlls.cpu().tolist())
    for items in records:
        csv_logger.info([item for item in items])

    # check NLL for target smiles if provided
    if model_type == "Mol2Mol":
        target_smiles = []
        target_smiles_path = parameters.get("target_smiles_path", "")
        if target_smiles_path:
            with open(target_smiles_path) as f:
                target_smiles = [line.rstrip("\n") for line in f]
        if len(target_smiles) > 0:
            input, target, tanimoto, nlls = sampler.check_nll(input_smilies, target_smiles)
            file = os.path.join(os.path.dirname(target_smiles_path), "target_nll.csv")
            csv_logger_target_nll = setup_logger(
                name="csv",
                filename=file,
                formatter=CsvFormatter(),
                propagate=False,
                level="INFO",
            )
            csv_logger_target_nll.info(HEADERS[model_type])
            records = zip(input, target, tanimoto, nlls)
            for items in records:
                csv_logger_target_nll.info([item for item in items])


def filter_valid(sampled: SampleBatch) -> SampleBatch:
    """Filter out valid SMILES and associated data, which is unique as well

    :param sampled:
    :return:
    """

    state = np.array(sampled.states)
    mask_idx = np.nonzero(state == SmilesState.VALID)[0]

    # For Reinvent, items1 is None
    items1 = list(np.array(sampled.items1)[mask_idx]) if sampled.items1 else None
    items2 = list(np.array(sampled.items2)[mask_idx])

    nlls = sampled.nlls[mask_idx]

    smilies = list(np.array(sampled.smilies)[mask_idx])
    states = np.array([SmilesState.VALID] * len(mask_idx))

    return SampleBatch(items1, items2, nlls, smilies, states)
