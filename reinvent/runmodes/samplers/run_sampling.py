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
from reinvent.utils import get_reporter, read_smiles_csv_file, get_tokens_from_vocabulary
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.models.model_factory.sample_batch import SampleBatch, SmilesState
from reinvent.chemistry import conversions
from reinvent.scoring import Scorer
from reinvent.runmodes.samplers.pepinvent import PepinventSampler
from reinvent_plugins.normalizers.rdkit_smiles import normalize
from .validation import SamplingConfig

logger = logging.getLogger(__name__)

HEADERS = {
    "Reinvent": ("SMILES", "SMILES_state", "NLL"),
    "Libinvent": ("SMILES", "SMILES_state", "Scaffold", "R-groups", "NLL"),
    "Linkinvent": ("SMILES", "SMILES_state", "Warheads", "Linker", "NLL"),
    "LibinventTransformer": ("SMILES", "SMILES_state", "Scaffold", "R-groups", "NLL"),
    "LinkinventTransformer": ("SMILES", "SMILES_state", "Warheads", "Linker", "NLL"),
    "Mol2Mol": ("SMILES", "SMILES_state", "Input_SMILES", "Tanimoto", "NLL"),
    "Pepinvent": ("SMILES", "SMILES_state", "Masked_input_peptide", "Fillers", "NLL"),
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
        allowed_tokens = get_tokens_from_vocabulary(adapter.vocabulary)
        input_smilies = read_smiles_csv_file(smiles_input_filename, 0, allowed_tokens)
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
        n = len(sampled.smilies)
        sampled = filter_valid(sampled)
        logger.info(f"Removed {n - len(sampled.smilies)} invalid SMILES")

    if config.filter:
        n = len(sampled.smilies)
        sampled = filter_by_pattern(sampled, config.filter.smarts)
        logger.info(f"Removed {n - len(sampled.smilies)} SMILES matching a filter pattern")

    records = None
    nlls = [round(nll, 2) for nll in sampled.nlls.cpu().tolist()]
    states = [state.value for state in sampled.states]

    if model_type == "Reinvent":
        records = zip(sampled.smilies, states, nlls)
    elif model_type in FRAGMENT_GENERATORS:
        records = zip(sampled.smilies, states, sampled.items1, sampled.items2, nlls)

        if model_type == "Pepinvent":
            filler_headers, filler_columns = PepinventSampler.split_fillers(sampled)
            HEADERS[model_type] += tuple(filler_headers)
            records = zip(sampled.smilies, states, sampled.items1, sampled.items2, nlls, *filler_columns)
    elif model_type == "Mol2Mol":
        if parameters.unique_molecules:
            mask_idx = np.nonzero(state == SmilesState.VALID)[0]
            scores = [scores[i] for i in mask_idx]
        records = zip(sampled.smilies, states, sampled.items1, scores, nlls)

        if parameters.target_smiles_path:
            allowed_tokens = get_tokens_from_vocabulary(adapter.vocabulary)
            target_smilies = read_smiles_csv_file(parameters.target_smiles_path, 0, allowed_tokens)

            if len(target_smilies) > 0:
                input, target, tanimoto, nlls = sampler.check_nll(input_smilies, target_smilies)

                with open(parameters.target_nll_file, "w") as fh:
                    write_csv(fh, HEADERS[model_type], zip(input, target, tanimoto, nlls))

    with open(smiles_output_filename, "w") as fh:
        write_csv(fh, HEADERS[model_type], records)


def filter_valid(sampled: SampleBatch) -> SampleBatch:
    """Filter out valid SMILES and associated data, which is unique as well

    :param sampled: sample batch
    :returns: filtered sample batch
    """

    state = np.array(sampled.states)
    mask_idx = state == SmilesState.VALID

    return filter_wanted_samples(sampled, mask_idx)


def filter_by_pattern(sampled: SampleBatch, patterns: list) -> SampleBatch:
    """Filter out SMILES and associated data, which is unique as well

    :param sampled: sample batch
    :return:
    """

    config = dict(
        type="geometric_mean",
        filename=None,
        component=[
            {"custom_alerts": {"endpoint": [{"name": "Alerts", "params": {"smarts": patterns}}]}}
        ],
    )

    custom_alerts = Scorer(config)
    size = len(sampled.smilies)
    score_results = custom_alerts(sampled.smilies, np.ones(size), np.ones(size), None)
    mask_idx = np.where(score_results.total_scores == 0, False, True)

    return filter_wanted_samples(sampled, mask_idx)


def filter_wanted_samples(sampled: SampleBatch, mask_idx: np.array):
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
