#!/usr/bin/env python
"""Main entry point into Reinvent."""

from __future__ import annotations
import os
import sys
import argparse
from dotenv import load_dotenv
import platform
import getpass
import random
import logging
import datetime
import subprocess as sp
from typing import List, Optional

SYSTEM = platform.system()

if SYSTEM != "Windows":
    import resource  # Unix only

from rdkit import rdBase, RDLogger
import numpy as np
import rdkit
import torch

from reinvent import version, runmodes, config_parse, setup_logger
from reinvent.runmodes.utils import set_torch_device
from reinvent.runmodes.reporter.remote import setup_reporter

INPUT_FORMAT_CHOICES = ("toml", "json")
RDKIT_CHOICES = ("all", "error", "warning", "info", "debug")
LOGLEVEL_CHOICES = tuple(level.lower() for level in logging._nameToLevel.keys())
VERSION_STR = f"{version.__progname__} {version.__version__} {version.__copyright__}"
OVERWRITE_STR = "Overwrites setting in the configuration file"
RESPONDER_TOKEN = "RESPONDER_TOKEN"

rdBase.DisableLog("rdApp.*")


def enable_rdkit_log(levels: List[str]):
    """Enable logging messages from RDKit for a specific logging level.

    :param levels: the specific level(s) that need to be silenced
    """

    if "all" in levels:
        RDLogger.EnableLog("rdApp.*")
        return

    for level in levels:
        RDLogger.EnableLog(f"rdApp.{level}")


def get_cuda_driver_version() -> Optional[str]:
    """Get the CUDA driver version via modinfo if possible.

    This is for Linux only.

    :returns: driver version or None
    """

    # Alternative
    # result = sp.run(["/usr/bin/nvidia-smi"], shell=False, capture_output=True)
    # if "Driver Version:" in str_line:
    #    version = str_line.split()[5]

    try:
        result = sp.run(["/sbin/modinfo", "nvidia"], shell=False, capture_output=True)
    except Exception:
        return

    for line in result.stdout.splitlines():
        str_line = line.decode()

        if str_line.startswith("version:"):
            cuda_driver_version = str_line.split()[1]
            return cuda_driver_version


def set_seed(seed: int):
    """Set global seed for reproducibility

    :param seed: the seed to initialize the random generators
    """

    if seed is None:
        return

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"{version.__progname__}: a molecular design "
        f"tool for de novo design, "
        "scaffold hopping, R-group replacement, linker design, molecule "
        "optimization, and others",
        epilog=f"{VERSION_STR}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_filename",
        nargs="?",
        default=None,
        metavar="FILE",
        type=os.path.abspath,
        help="Input configuration file with runtime parameters",
    )

    parser.add_argument(
        "-f",
        "--config-format",
        metavar="FORMAT",
        choices=INPUT_FORMAT_CHOICES,
        default="toml",
        help=f"File format of the configuration file: {', '.join(INPUT_FORMAT_CHOICES)}",
    )

    parser.add_argument(
        "-d",
        "--device",
        metavar="DEV",
        default=None,
        help=f"Device to run on: cuda, cpu. {OVERWRITE_STR}.",
    )

    parser.add_argument(
        "-l",
        "--log-filename",
        metavar="FILE",
        default=None,
        type=os.path.abspath,
        help=f"File for logging information, otherwise writes to stderr.",
    )

    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=LOGLEVEL_CHOICES,
        default="info",
        help=f"Enable this and 'higher' log levels: {', '.join(LOGLEVEL_CHOICES)}.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        metavar="N",
        type=int,
        default=None,
        help="Sets the random seeds for reproducibility",
    )

    parser.add_argument(
        "--dotenv-filename",
        metavar="FILE",
        default=None,
        type=os.path.abspath,
        help=f"Dotenv file with environment setup needed for some scoring components. "
        "By default the one from the installation directory will be loaded.",
    )

    parser.add_argument(
        "--enable-rdkit-log-levels",
        metavar="LEVEL",
        choices=RDKIT_CHOICES,
        nargs="+",
        help=f"Enable specific RDKit log levels: {', '.join(RDKIT_CHOICES)}.",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"{VERSION_STR}.",
    )

    return parser.parse_args()


def setup_responder(config):
    """Setup for remote monitor

    :param config: configuration
    """

    endpoint = config.get("endpoint", False)

    if not endpoint:
        return

    token = os.environ.get(RESPONDER_TOKEN, None)
    setup_reporter(endpoint, token)


def main():
    """Simple entry point into Reinvent's run modes."""

    args = parse_command_line()

    dotenv_loaded = load_dotenv(args.dotenv_filename)  # set up the environment for scoring

    reader = getattr(config_parse, f"read_{args.config_format}")
    input_config = reader(args.config_filename)

    if args.enable_rdkit_log_levels:
        enable_rdkit_log(args.enable_rdkit_log_levels)

    run_type = input_config["run_type"]
    runner = getattr(runmodes, f"run_{run_type}")
    logger = setup_logger(
        name=__package__, level=args.log_level.upper(), filename=args.log_filename
    )

    have_version = input_config.get("version", version.__config_version__)

    if have_version < version.__config_version__:
        msg = f"Need at least version 4.  Input file is for version {have_version}."
        logger.fatal(msg)
        raise RuntimeError(msg)

    logger.info(
        f"Started {version.__progname__} {version.__version__} {version.__copyright__} on "
        f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
    )

    logger.info(f"Command line: {' '.join(sys.argv)}")

    if dotenv_loaded:
        logger.info("Environment loaded from dotenv file")

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
        logger.info(f"GPU memory: {free_memory // 1024**2} MiB free, "
                    f"{total_memory // 1024**2} MiB total")
    else:
        logger.info(f"Using CPU {platform.processor()}")

    seed = input_config.get("seed", None)

    if args.seed is not None:
        set_seed(seed)
        logger.info(f"Set seed for all random generators to {seed}")

    tb_logdir = None

    if "tb_logdir" in input_config:
        tb_logdir = os.path.abspath(input_config["tb_logdir"])
        logger.info(f"Writing TensorBoard summary to {tb_logdir}")

    if "json_out_config" in input_config:
        json_out_config = os.path.abspath(input_config["json_out_config"])
        logger.info(f"Writing JSON config file to {json_out_config}")
        config_parse.write_json(input_config, json_out_config)

    responder_config = None

    if "responder" in input_config:
        setup_responder(input_config["responder"])
        responder_config = input_config["responder"]
        logger.info(f"Using remote monitor endpoint {input_config['responder']['endpoint']} "
                    f"with frequency {input_config['responder']['frequency']}")

    runner(input_config, actual_device, tb_logdir, responder_config)

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


if __name__ == "__main__":
    main()
