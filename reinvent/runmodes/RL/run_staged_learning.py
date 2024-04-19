"""Multi-stage learning with RL"""

from __future__ import annotations
import os
import logging
from typing import List, TYPE_CHECKING

import torch

from reinvent import config_parse, setup_logger, CsvFormatter
from reinvent.runmodes import Handler, RL, create_adapter
from reinvent.runmodes.dtos import ChemistryHelpers
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.RL import terminators, memories
from reinvent.runmodes.RL.data_classes import WorkPackage, ModelState
from reinvent.runmodes.utils import disable_gradients
from reinvent.scoring import Scorer
from reinvent.chemistry import Conversions
from reinvent.chemistry.library_design import (
    BondMaker,
    AttachmentPoints,
)

if TYPE_CHECKING:
    from reinvent.runmodes.RL import terminator_callable
    from reinvent.models import ModelAdapter

logger = logging.getLogger(__name__)


def setup_diversity_filter(config: dict, conversions, rdkit_smiles_flags: dict):
    """Setup of the diversity filter

    Basic setup of the diversity filter memory.  The parameters are from a
    dict, so the keys (parameters) are hard-coded here.

    :param config: config parameter specific to the filter
    :param conversions: chemistry conversions
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: the set up diversity filter
    """

    if config is None:
        return None

    memory_type = config["type"]

    if "type" in config:
        diversity_filter = getattr(memories, memory_type)
    else:
        return None

    logger.info(f"Using diversity filter {memory_type}")

    return diversity_filter(
        bucket_size=config.get("bucket_size", 20),
        minscore=config.get("minscore", 1.0),
        minsimilarity=config.get("minsimilarity", 0.4),
        penalty_multiplier=config.get("penalty_multiplier", 0.5),
        conversions=conversions,
        rdkit_smiles_flags=rdkit_smiles_flags,
    )


def setup_reward_strategy(config: dict, agent: ModelAdapter):
    """Setup the Reinforcement Learning reward strategy

    Basic parameter setup for RL learning including the reward function. The
    parameters are from a dict, so the keys (parameters) are hard-coded here.

    DAP has been found to be the best choice, see https://doi.org/10.1021/acs.jcim.1c00469.
    SDAP seems to have a smaller learning rate while the other two (MAULI, MASCOF)
    do not seem to bes useful at all.

    :param config: the configuration
    :param agent: the agent model network
    :return: the set up RL strategy
    """

    learning_rate = config["rate"]
    sigma = config["sigma"]  # determines how dominant the score is

    reward_strategy_str = config["type"]

    try:
        reward_strategy = getattr(RL, f"{reward_strategy_str}_strategy")
    except AttributeError:
        msg = f"Unknown reward strategy {reward_strategy_str}"
        logger.critical(msg)
        raise RuntimeError(msg)

    torch_optim = torch.optim.Adam(agent.get_network_parameters(), lr=learning_rate)
    learning_strategy = RL.RLReward(torch_optim, sigma, reward_strategy)

    logger.info(f"Using reward strategy {reward_strategy_str}")

    return learning_strategy


def setup_inception(config: dict, prior: ModelAdapter):
    """Setup inception memory

    :param config: the config specific to the inception memory
    :param prior: the prior network
    :return: the set up inception memory or None
    """

    smilies = []
    deduplicate = config.get("deduplicate", True)
    smilies_filename = config.get("smiles_file", None)

    if smilies_filename and os.path.exists(smilies_filename):
        smilies = config_parse.read_smiles_csv_file(smilies_filename, columns=0)

        if not smilies:
            msg = f"Inception SMILES could not be read from {smilies_filename}"
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.info(f"Inception SMILES read from {smilies_filename}")

    if not smilies:
        logger.info(f"No SMILES for inception. Populating from first sampled batch.")

    if deduplicate:
        logger.info("Global SMILES deduplication for inception memory")

    inception = memories.Inception(
        memory_size=config["memory_size"],
        sample_size=config["sample_size"],
        smilies=smilies,
        scoring_function=None,
        prior=prior,
        deduplicate=deduplicate,
    )

    logger.info(f"Using inception memory")

    return inception


def setup_scoring(config: dict) -> dict:
    """Update scoring component from file if requested

    :param config: scoring dictionary
    :returns: scoring dictionary
    """

    component_filename = config.get("filename", None)
    component_filetype = config.get("filetype", None)

    if component_filename and component_filetype:
        logger.info(f"Reading score components from {component_filename}")
        parser = getattr(config_parse, f"read_{component_filetype.lower()}")
        components_config = parser(component_filename)
        config.update(components_config)

    return config


def create_packages(reward_strategy: RL.RLReward, stages: list) -> List[WorkPackage]:
    """Create work packages

    Collect the stage parameters and build a work package for each stage.  The
    parameters are from a dict, so the keys (parameters) are hard-coded here.
    Each stage can define its own scoring function.

    :param reward_strategy: the reward strategy
    :param stages: the parameters for each work package
    :return: a list of work packages
    """
    packages = []

    for stage in stages:
        chkpt_filename = stage["chkpt_file"]

        scoring_config = setup_scoring(stage["scoring"])
        scoring_function = Scorer(scoring_config)

        max_score = stage.get("max_score", 1.0)
        min_steps = stage.get("min_steps", 1)
        max_steps = stage.get("max_steps", 10)  # hard limit

        terminator_param = stage.get("termination", "null")
        terminator_name = terminator_param.lower().title()

        try:
            terminator: terminator_callable = getattr(terminators, f"{terminator_name}Terminator")
        except KeyError:
            msg = f"Unknown termination criterion: {terminator_name}"
            logger.critical(msg)
            raise RuntimeError(msg)

        packages.append(
            WorkPackage(
                scoring_function,
                reward_strategy,
                max_steps,
                terminator(max_score, min_steps),
                chkpt_filename,
            )
        )

    return packages


def run_staged_learning(
    config: dict,
    device: torch.device,
    tb_logdir: str,
    responder_config: dict,
    *args,
    **kwargs,
):
    """Run Reinforcement Learning/Curriculum Learning

    RL can be run in multiple stages (CL) with different parameters, typically
    the scoring function.  A context manager ensures that a checkpoint file is
    written out should the program be terminated.

    :param config: the run configuration
    :param device: torch device
    :param tb_logdir: TensorBoard log directory
    :param responder_config: responder configuration
    """

    stages = config["stage"]
    num_stages = len(stages)
    logger.info(
        f"Starting {num_stages} {'stages' if num_stages> 1 else 'stage'} of Reinforcement Learning"
    )

    parameters = config["parameters"]

    # NOTE: The model files are a dictionary with model attributes from
    #       Reinvent and a set of tensors, each with an attribute for the
    #       device (CPU or GPU) and if gradients are required

    prior_model_filename = os.path.abspath(parameters["prior_file"])
    agent_model_filename = os.path.abspath(parameters["agent_file"])

    # NOTE: Inference mode means here that torch runs eval() on the network:
    #       switch off some specific layers (dropout, batch normal) but that
    #       does not affect autograd
    #       The gradients are switched off for the prior but the optimizer will
    #       not touch those anyway because we pass only the agent network to the
    #       optimizer, see above.
    adapter, _, model_type = create_adapter(prior_model_filename, "inference", device)
    prior = adapter
    disable_gradients(prior)

    rdkit_smiles_flags = dict(allowTautomers=True)

    if model_type in ["Mol2Mol", "LinkinventTransformer"]:  # Transformer-based models
        agent_mode = "inference"
        rdkit_smiles_flags.update(sanitize=True, isomericSmiles=True)
        rdkit_smiles_flags2 = dict(isomericSmiles=True)
    else:
        agent_mode = "training"
        rdkit_smiles_flags2 = dict()

    adapter, agent_save_dict, agent_model_type = create_adapter(
        agent_model_filename, agent_mode, device
    )
    agent = adapter

    if model_type != agent_model_type:
        msg = f"Inconsistent model types: prior is {model_type} agent is {agent_model_type}"
        logger.critical(msg)
        raise RuntimeError(msg)

    logger.info(f"Using generator {model_type}")
    logger.info(f"Prior read from {prior_model_filename}")
    logger.info(f"Agent read from {agent_model_filename}")

    try:
        smilies_filename = parameters["smiles_file"]
        smilies = config_parse.read_smiles_csv_file(smilies_filename, columns=0)
        logger.info(f"Input molecules/fragments read from file {smilies_filename}")
    except KeyError:  # optional for Reinvent
        smilies = None

    # The chemistry helpers are mostly static functions with little state (only
    # AttachmentPoints needs constants from TransformationTokens.
    # AttachmentPoints depends on Conversions and BondMaker on both
    # Conversions and AttachmentPoints
    chemistry = ChemistryHelpers(
        Conversions(),  # Lib/LinkInvent, Mol2Mol
        BondMaker(),  # LibInvent
        AttachmentPoints(),  # Lib/LinkInvent
    )

    sampler, _ = setup_sampler(model_type, parameters, agent, chemistry)
    reward_strategy = setup_reward_strategy(config["learning_strategy"], agent)
    df_section = config.get("diversity_filter", None)

    if parameters["use_checkpoint"] and "staged_learning" in agent_save_dict:
        logger.info(f"Using diversity filter from {agent_model_filename}")
        diversity_filter = agent_save_dict["staged_learning"]["diversity_filter"]
    else:
        diversity_filter = setup_diversity_filter(
            df_section, chemistry.conversions, rdkit_smiles_flags2
        )

    purge_diversity_filter = parameters.get("purge_memories", False)

    if purge_diversity_filter:
        logger.info("Purging diversity filter memories after each stage")
    else:
        logger.info("Diversity filter memories are retained between stages")

    inception = None

    # Inception only set up for the very first step
    if "inception" in config and model_type == "Reinvent":
        inception = setup_inception(config["inception"], prior)

    if not inception and model_type == "Reinvent":
        logger.warning("Inception disabled but may speed up convergence")

    state = ModelState(agent, diversity_filter)
    packages = create_packages(reward_strategy, stages)

    summary_csv_prefix = parameters.get("summary_csv_prefix", "summary")

    # FIXME: is there a sensible default, this is only needed by Mol2Mol
    distance_threshold = parameters.get("distance_threshold", 99999)

    model_learning = getattr(RL, f"{model_type}Learning")

    with Handler() as handler:
        for run, package in enumerate(packages):
            stage_no = run + 1
            csv_filename = f"{summary_csv_prefix}_{stage_no}.csv"

            setup_logger(
                name="csv",
                filename=csv_filename,
                formatter=CsvFormatter(),
                propagate=False,
                level="INFO",
            )

            logdir = f"{tb_logdir}_{run}" if tb_logdir else None

            logger.info(f"Writing tabular data for stage to {csv_filename}")
            logger.info(f"Starting stage {stage_no} <<<")

            optimize = model_learning(
                max_steps=package.max_steps,
                prior=prior,
                state=state,
                scoring_function=package.scoring_function,
                reward_strategy=package.learning_strategy,
                sampling_model=sampler,
                smilies=smilies,
                distance_threshold=distance_threshold,
                rdkit_smiles_flags=rdkit_smiles_flags,
                inception=inception,
                chemistry=chemistry,
                responder_config=responder_config,
                tb_logdir=logdir,
            )

            if device.type == "cuda" and torch.cuda.is_available():
                free_memory, total_memory = torch.cuda.mem_get_info()
                free_memory //= 1024**2
                used_memory = total_memory // 1024**2 - free_memory
                logger.info(f"Current GPU memory usage: {used_memory} MiB used, {free_memory} MiB free")

            handler.out_filename = package.out_state_filename
            handler.register_callback(optimize.get_state_dict)

            if inception:
                inception.update_scoring_function(package.scoring_function)

            terminate = optimize(package.terminator)
            state = optimize.state

            if purge_diversity_filter:
                logger.info(f"Purging diversity filter memories in stage {stage_no}")
                state.diversity_filter.purge_memories()

            handler.save()

            if terminate:
                logger.warning(
                    f"Maximum number of steps of {package.max_steps} reached in stage "
                    f"{stage_no}. Terminating all stages."
                )
                break

            logger.info(f"Finished stage {stage_no} >>>")
