"""Sample SMILES from a model

LibInvent, LinkInvent and Mol2Mol need input SMILES while Reinvent does not.
The output is a list of SMILES including the fragments where applicable and
the negative log likelihood if requested.
"""

__all__ = ["run_sampling"]
import logging

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from reinvent.runmodes import create_adapter
from reinvent.runmodes.samplers.reports import (
    SamplingTBReporter,
    SamplingRemoteReporter,
)
from reinvent.utils import get_reporter, read_smiles_csv_file
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState
from reinvent.chemistry import conversions
from reinvent_plugins.normalizers.rdkit_smiles import normalize
from .validation import SamplingConfig

logger = logging.getLogger(__name__)

HEADERS = {
    "Reinvent": ("SMILES", "NLL"),
    "Libinvent": ("SMILES", "Scaffold", "R-groups", "NLL"),
    "Linkinvent": ("SMILES", "Warheads", "Linker", "NLL"),
    "LibinventTransformer": ("SMILES", "Scaffold", "R-groups", "NLL"),
    "LinkinventTransformer": ("SMILES", "Warheads", "Linker", "NLL"),
    "Mol2Mol": ("SMILES", "Input_SMILES", "Tanimoto", "NLL"),
    "Pepinvent": ("SMILES", "Masked_input_peptide", "Fillers", "NLL"),
}

FRAGMENT_GENERATORS = [
    "Libinvent",
    "Linkinvent",
    "LinkinventTransformer",
    "LibinventTransformer",
    "Pepinvent",
]


def run_sampling(
    input_config: dict, device, tb_logdir: str, write_config: str = None, *args, **kwargs
):
    """Sampling run setup"""

    logger.info("Starting Sampling")

    config = SamplingConfig(**input_config)
    parameters = config.parameters
    smiles_output_filename = parameters.output_file

    agent_model_filename = parameters.model_file
    adapter, _, model_type = create_adapter(agent_model_filename, "inference", device)

    logger.info(f"Using generator {model_type}")
    logger.info(f"Writing sampled SMILES to CSV file {smiles_output_filename}")

    # number of smiles to be generated for each input; consistent with batch_size parameter as used in RL
    # different from batch size used in dataloader which affect cuda memory
    params = parameters.dict()
    params["batch_size"] = parameters.num_smiles
    sampler, batch_size = setup_sampler(model_type, params, adapter)
    sampler.unique_sequences = False

    try:
        smiles_input_filename = parameters.smiles_file
    except KeyError:
        smiles_input_filename = None

    input_smilies = None
    num_input_smilies = 1

    if smiles_input_filename:
        input_smilies = read_smiles_csv_file(smiles_input_filename, columns=0)
        num_input_smilies = len(input_smilies)

    num_total_smilies = parameters.num_smiles * num_input_smilies

    logger.info(f"Sampling {num_total_smilies} SMILES from model {agent_model_filename}")

    # NOTE: for beamsearch the batch size determines the beam size
    if model_type == "Mol2Mol" and parameters.sample_strategy == "beamsearch":
        if parameters.num_smiles > 300:
            logger.warning(f"Sampling with beam search may be very slow")

    if callable(write_config):
        write_config(config.model_dump())

    with torch.no_grad():
        sampled = sampler.sample(input_smilies)

    # FIXME: remove atom map numbers from SMILES in chemistry code
    if model_type == "Libinvent":
        sampled.smilies = normalize(sampled.smilies, keep_all=True)

    kwargs = {}
    scores = [-1] * len(sampled.items2)
    state = np.array(sampled.states)

    if model_type == "Mol2Mol":
        # compute Tanimoto similarity between generated compounds and input compounds; return largest
        valid_mols, valid_idxs = conversions.smiles_to_mols_and_indices(sampled.items2)
        valid_scores = sampler.calculate_tanimoto(input_smilies, sampled.items2)

        for i, j in enumerate(valid_idxs):
            scores[j] = valid_scores[i]

        kwargs = {"Tanimoto": scores}

    reporters = setup_reporters(tb_logdir)

    for reporter in reporters:
        reporter.submit(sampled, **kwargs)

    if parameters.unique_molecules:
        sampled = filter_valid(sampled)

    records = None
    nlls = [round(nll, 2) for nll in sampled.nlls.cpu().tolist()]

    if model_type == "Reinvent":
        records = zip(sampled.smilies, nlls)
    elif model_type in FRAGMENT_GENERATORS:
        records = zip(sampled.smilies, sampled.items1, sampled.items2, nlls)
    elif model_type == "Mol2Mol":
        if parameters.unique_molecules:
            mask_idx = np.nonzero(state == SmilesState.VALID)[0]
            scores = [scores[i] for i in mask_idx]
        records = zip(sampled.smilies, sampled.items1, scores, nlls)

        if parameters.target_smiles_path:
            target_smilies = read_smiles_csv_file(parameters.target_smiles_path, columns=0)

            if len(target_smilies) > 0:
                input, target, tanimoto, nlls = sampler.check_nll(input_smilies, target_smilies)

                with open(parameters.target_nll_file, "w") as fh:
                    write_csv(fh, HEADERS[model_type], zip(input, target, tanimoto, nlls))

    with open(smiles_output_filename, "w") as fh:
        write_csv(fh, HEADERS[model_type], records)


def filter_valid(sampled: SampleBatch) -> SampleBatch:
    """Filter out valid SMILES and associated data, which is unique as well

    :param sampled:
    :return:
    """

    state = np.array(sampled.states)
    mask_idx = state == SmilesState.VALID

    # For Reinvent, items1 is None
    items1 = list(np.array(sampled.items1)[mask_idx]) if sampled.items1 else None
    items2 = list(np.array(sampled.items2)[mask_idx])

    nlls = sampled.nlls[mask_idx]

    smilies = list(np.array(sampled.smilies)[mask_idx])
    states = sampled.states[mask_idx]

    return SampleBatch(items1, items2, nlls, smilies, states)


def write_csv(fh, headers, records):
    fh.write(",".join(headers) + "\n")

    for items in records:
        line = ",".join([str(item) for item in items])
        fh.write(f"{line}\n")


def setup_reporters(tb_logdir):
    """Set up reporters

    Choices: CSV (SMILES), TensorBoard, remote
    """

    reporters = []
    remote_reporter = get_reporter()
    tb_reporter = None

    if tb_logdir:
        tb_reporter = SummaryWriter(log_dir=tb_logdir)

    for kls, reporter in (SamplingTBReporter, tb_reporter), (
        SamplingRemoteReporter,
        remote_reporter,
    ):
        if reporter:
            reporters.append(kls(reporter))

    return reporters
