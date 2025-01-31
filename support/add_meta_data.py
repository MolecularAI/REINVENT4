#!/bin/env python3
"""Add meta data to REINVENT model files"""

import os
import sys
import uuid
import time
import logging
import argparse
from typing import TextIO

import torch
import xxhash

from reinvent.models import meta_data
from reinvent.models.meta_data import update_model_data, check_valid_hash


def setup_logger(
    filename: str = None,
    stream: TextIO = sys.stderr,
    propagate: bool = True,
    level=logging.INFO,
):
    """Setup simple logging

    :param filename: optional filename for logging output
    :param stream: the output stream
    :param propagate: whether to propagate to higher level loggers
    :param level: logging level
    :returns: the newly set up logger
    """

    logging.captureWarnings(True)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if filename:
        handler = logging.FileHandler(filename, mode="w+")
    else:
        handler = logging.StreamHandler(stream)

    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s <%(levelname)-4.4s> %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = propagate

    return logger


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"{sys.argv[0]}: add metadata to a REINVENT model file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_model_file",
        default=None,
        type=os.path.abspath,
        help="Input model file to be added with metadata",
    )

    parser.add_argument(
        "output_model_file",
        default=None,
        type=os.path.abspath,
        help="Output model file with updated metadata",
    )

    parser.add_argument(
        "-s",
        "--source",
        default="",
        help=f"Optional data source for model",
    )
    parser.add_argument(
        "-c",
        "--comment",
        default="",
        help=f"Optional user comment",
    )

    return parser.parse_args()


def main():
    logger = setup_logger()
    args = parse_command_line()

    device = torch.device("cpu")

    logger.info(f"Reading {args.input_model_file}")
    model = torch.load(args.input_model_file, map_location=device, weights_only=False)

    if "metadata" not in model:
        model["metadata"] = meta_data.ModelMetaData(
            creation_date=time.time(),
            hash_id=None,
            hash_id_format="",
            model_id=uuid.uuid4().hex,
            origina_data_source=args.source,
        )

    new_model = update_model_data(model, args.comment, write_update=False)

    logger.info(f"Writing {args.output_model_file}")
    torch.save(new_model, args.output_model_file)

    model = torch.load(args.output_model_file, map_location=device, weights_only=False)
    valid = check_valid_hash(model)

    if not valid:
        logger.error(f"Invalid hash")


if __name__ == "__main__":
    main()
