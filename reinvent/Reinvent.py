#!/usr/bin/env python
"""Main entry point into Reinvent."""

from __future__ import annotations
import os
import sys
from dotenv import load_dotenv, find_dotenv
import platform
import getpass
import datetime
from typing import Any

from reinvent.utils import (
    parse_command_line,
    get_cuda_driver_version,
    set_seed,
    extract_sections,
    write_json_config,
    enable_rdkit_log,
    setup_responder,
    config_parse,
)

SYSTEM = platform.system()

if SYSTEM != "Windows":
    import resource  # Unix only

from rdkit import rdBase
import rdkit
import torch

from reinvent import version, runmodes
from reinvent.utils import setup_logger
from reinvent.runmodes.utils import set_torch_device
from reinvent.runmodes.handler import StageInterruptedControlled
from .validation import ReinventConfig


rdBase.DisableLog("rdApp.*")


def main(args: Any):
    """Simple entry point into Reinvent's run modes.

    :param args: arguments object, can be argparse.Namespace or any other class
    """

    logger = setup_logger(
        name=__package__, level=args.log_level.upper(), filename=args.log_filename
    )

    logger.info(
        f"Started {version.__progname__} {version.__version__} {version.__copyright__} on "
        f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
    )

    logger.info(f"Command line: {' '.join(sys.argv)}")

    dotenv_loaded = load_dotenv(args.dotenv_filename)  # set up the environment for scoring

    ext = None

    if args.config_filename:
        ext = args.config_filename.suffix

    if ext in (f".{e}" for e in config_parse.INPUT_FORMAT_CHOICES):
        fmt = ext[1:]
    else:
        fmt = args.config_format

    logger.info(f"Reading run configuration from {args.config_filename} using format {fmt}")
    input_config = config_parse.read_config(args.config_filename, fmt)
    val_config = ReinventConfig(**input_config)

    if args.enable_rdkit_log_levels:
        enable_rdkit_log(args.enable_rdkit_log_levels)

    run_type = input_config["run_type"]
    runner = getattr(runmodes, f"run_{run_type}")

    have_version = input_config.get("version", version.__config_version__)

    if have_version < version.__config_version__:
        msg = f"Need at least version 4.  Input file is for version {have_version}."
        logger.fatal(msg)
        raise RuntimeError(msg)

    if dotenv_loaded:
        if args.dotenv_filename:
            filename = args.dotenv_filename
        else:
            filename = find_dotenv()

        logger.info(f"Environment loaded from dotenv file {filename}")

    logger.info(f"User {getpass.getuser()} on host {platform.node()}")
    logger.info(f"Python version {platform.python_version()}")
    logger.info(f"PyTorch version {torch.__version__}, " f"git {torch.version.git_version}")
    logger.info(f"PyTorch compiled with CUDA version {torch.version.cuda}")
    logger.info(f"RDKit version {rdkit.__version__}")
    logger.info(f"Platform {platform.platform()}")

    if cuda_driver_version := get_cuda_driver_version():
        logger.info(f"CUDA driver version {cuda_driver_version}")

    logger.info(f"Number of PyTorch CUDA devices {torch.cuda.device_count()}")

    if "use_cuda" in input_config:
        logger.warning("'use_cuda' is deprecated, use 'device' instead")

    device = input_config.get("device", None)

    if not device:
        use_cuda = input_config.get("use_cuda", True)

        if use_cuda:
            device = "cuda:0"

    actual_device = set_torch_device(args.device, device)

    if actual_device.type == "cuda":
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Using CUDA device:{current_device} {device_name}")

        free_memory, total_memory = torch.cuda.mem_get_info()
        logger.info(
            f"GPU memory: {free_memory // 1024**2} MiB free, "
            f"{total_memory // 1024**2} MiB total"
        )
    else:
        logger.info(f"Using CPU {platform.processor()}")

    seed = args.seed or input_config.get("seed", None)

    if seed is not None:
        set_seed(seed)
        logger.info(f"Set seed for all random generators to {seed}")

    tb_logdir = input_config.get("tb_logdir", None)

    if tb_logdir:
        tb_logdir = os.path.abspath(tb_logdir)
        logger.info(f"Writing TensorBoard summary to {tb_logdir}")

    write_config = None

    if "json_out_config" in input_config:
        json_out_config = os.path.abspath(input_config["json_out_config"])
        logger.info(f"Writing JSON config file to {json_out_config}")
        write_config = write_json_config(val_config.model_dump(), json_out_config)

    responder_config = input_config.get("responder", None)

    if responder_config:
        setup_responder(responder_config)
        logger.info(
            f"Using remote monitor endpoint {input_config['responder']['endpoint']} "
            f"with frequency {input_config['responder']['frequency']}"
        )

    try:
        runner(
            input_config=extract_sections(input_config),
            device=actual_device,
            tb_logdir=tb_logdir,
            responder_config=responder_config,
            write_config=write_config,
        )
    except StageInterruptedControlled as e:
        logger.info(f"Requested to terminate: {e.args[0]}")

    if SYSTEM != "Windows":
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem = 0

        if SYSTEM == "Linux":
            peak_mem = maxrss / 1024
        elif SYSTEM == "Darwin":  # MacOSX
            peak_mem = maxrss / 1024**2

        if peak_mem:
            logger.info(f"Peak main memory usage: {peak_mem:.3f} MiB")

    logger.info(
        f"Finished {version.__progname__} on {datetime.datetime.now().strftime('%Y-%m-%d')}"
    )


def main_script():
    """Main entry point from the command line"""

    args = parse_command_line()
    main(args)


if __name__ == "__main__":
    main_script()
