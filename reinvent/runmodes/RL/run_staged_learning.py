"""Multi-stage learning with RL"""

from __future__ import annotations
import logging

import torch

from reinvent.utils import setup_logger, CsvFormatter, config_parse, get_tokens_from_vocabulary
from reinvent.runmodes import Handler, RL, create_adapter
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.RL.data_classes import ModelState
from reinvent.runmodes.utils import disable_gradients
from .setup import create_packages, setup_diversity_filter, setup_intrinsic_penalty, setup_inception, setup_reward_strategy
from .validation import RLConfig

logger = logging.getLogger(__name__)

TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer", "LibinventTransformer", "Pepinvent"]


def run_staged_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str,
    responder_config: dict,
    write_config: str = None,
    *args,
    **kwargs,
):
    """Run Reinforcement Learning/Curriculum Learning

    RL can be run in multiple stages (CL) with different parameters, typically
    the scoring function.  A context manager ensures that a checkpoint file is
    written out should the program be terminated.

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: TensorBoard log directory
    :param responder_config: responder configuration
    :param write_config: callable to write config
    """

    config = RLConfig(**input_config)
    stages = config.stage
    num_stages = len(stages)
    logger.info(
        f"Starting {num_stages} {'stages' if num_stages> 1 else 'stage'} of Reinforcement Learning"
    )

    parameters = config.parameters

    # NOTE: The model files are a dictionary with model attributes from
    #       Reinvent and a set of tensors, each with an attribute for the
    #       device (CPU or GPU) and if gradients are required

    prior_model_filename = parameters.prior_file
    agent_model_filename = parameters.agent_file

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

    if model_type in TRANSFORMERS:  # Transformer-based models
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

    smilies = None

    if parameters.smiles_file:
        allowed_tokens = get_tokens_from_vocabulary(agent.vocabulary)
        smilies = config_parse.read_smiles_csv_file(parameters.smiles_file, 0, allowed_tokens)
        logger.info(f"Input molecules/fragments read from file {parameters.smiles_file}")

    sampler, _ = setup_sampler(model_type, parameters.dict(), agent)
    reward_strategy = setup_reward_strategy(config.learning_strategy, agent)

    global_df_only = False
    intrinsic_penalty = None

    if parameters.use_checkpoint and "staged_learning" in agent_save_dict:
        logger.info(f"Using diversity filter from {agent_model_filename}")
        diversity_filter = agent_save_dict["staged_learning"]["diversity_filter"]
    elif config.diversity_filter:
        diversity_filter = setup_diversity_filter(config.diversity_filter, rdkit_smiles_flags2)
        global_df_only = True
    elif config.intrinsic_penalty:
        intrinsic_penalty = setup_intrinsic_penalty(
            config.intrinsic_penalty,
            device,
            prior_model_filename,
            rdkit_smiles_flags2,
        )    

    if parameters.purge_memories:
        logger.info("Purging diversity filter memories after each stage")
    else:
        logger.info("Diversity filter memories are retained between stages")

    inception = None

    # Inception only set up here for the very first step
    if config.inception:  # and model_type == "Reinvent":
        inception = setup_inception(config.inception, prior)

    if inception is None and model_type == "Reinvent":
        logger.warning("Inception disabled but may speed up convergence")

    packages = create_packages(reward_strategy, stages, rdkit_smiles_flags2)

    summary_csv_prefix = parameters.summary_csv_prefix

    # FIXME: is there a sensible default, this is only needed by Mol2Mol
    distance_threshold = parameters.distance_threshold

    model_learning = getattr(RL, f"{model_type}Learning")

    if callable(write_config):
        write_config(config.model_dump())

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

            if global_df_only:  # global DF always overwrites stage DFs
                state = ModelState(agent, diversity_filter)
                logger.debug(f"Using global DF")
            else:
                state = ModelState(agent, package.diversity_filter)
                logger.debug(f"Using stage DF")

            optimize = model_learning(
                max_steps=package.max_steps,
                max_smiles=package.max_smiles,
                stage_no=stage_no,
                prior=prior,
                state=state,
                scoring_function=package.scoring_function,
                reward_strategy=package.learning_strategy,
                sampling_model=sampler,
                smilies=smilies,
                distance_threshold=distance_threshold,
                rdkit_smiles_flags=rdkit_smiles_flags,
                inception=inception,
                responder_config=responder_config,
                tb_logdir=logdir,
                tb_isim=parameters.tb_isim,
                intrinsic_penalty=intrinsic_penalty,
            )

            if hasattr(torch, device.type) and device.type != "cpu":
                gpu = getattr(torch, device.type)
                free_memory, total_memory = gpu.mem_get_info()
                free_memory //= 1024**2
                used_memory = total_memory // 1024**2 - free_memory
                logger.info(
                    f"Current GPU memory usage: {used_memory} MiB used, {free_memory} MiB free"
                )

            handler.out_filename = package.out_state_filename
            handler.register_callback(optimize.get_state_dict)

            if inception is not None:
                inception.update(package.scoring_function)

            terminate = optimize(package.terminator)
            state = optimize.state

            if state.diversity_filter and parameters.purge_memories:
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
