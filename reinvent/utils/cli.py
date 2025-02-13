"""Command line setup

FIXME: replace with dataclass and automatic conversion to argparse?
"""

from __future__ import annotations

__all__ = ["parse_command_line"]
import argparse
import os
from pathlib import Path
import logging

from reinvent import version
from reinvent.utils import config_parse

RDKIT_CHOICES = ("all", "error", "warning", "info", "debug")
LOGLEVEL_CHOICES = tuple(level.lower() for level in logging._nameToLevel.keys())
VERSION_STR = f"{version.__progname__} {version.__version__} {version.__copyright__}"
OVERWRITE_STR = "Overwrites setting in the configuration file"


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
        type=lambda fn: Path(fn).resolve(),
        help="Input configuration file with runtime parameters",
    )

    parser.add_argument(
        "-f",
        "--config-format",
        metavar="FORMAT",
        choices=config_parse.INPUT_FORMAT_CHOICES,
        default="toml",
        help=f"File format of the configuration file: {', '.join(config_parse.INPUT_FORMAT_CHOICES)}.  This can be used to force a specific format.  By default the format is derived from the file extension.",
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
        help=f"Sets the random seeds for reproducibility. {OVERWRITE_STR}.",
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
