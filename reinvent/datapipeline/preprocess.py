#!/usr/bin/env python
"""Tokenize SMILES

Assumes correct SMILES syntax e.g. that elements other than the basic ones are
in brackets.
"""
import os
import argparse
import pathlib
import json
import datetime
import multiprocessing as mp
from collections import Counter
import logging
import logging.handlers

import tomli
from rdkit import rdBase
import polars as pl
from tqdm import tqdm

from reinvent.datapipeline.filters import RegexFilter, SMILES_TOKENS_REGEX, RDKitFilter, elements
from reinvent.datapipeline.logger import setup_sp_logger, setup_mp_logger, logging_listener
from reinvent.datapipeline.validation import DPLConfig
from reinvent.datapipeline import normalizer


rdBase.DisableLog("rdApp.*")


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"Preprocess SMILES file from CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_filename",
        default=None,
        metavar="FILE",
        type=lambda p: pathlib.Path(p).absolute(),
        help="Input TOML configuration file with runtime parameters",
    )

    parser.add_argument(
        "-l",
        "--log-filename",
        metavar="FILE",
        default=None,
        type=lambda p: pathlib.Path(p).absolute(),
        help=f"File for logging information, otherwise writes to stderr.",
    )

    return parser.parse_args()


def main(args):

    with open(args.config_filename, "rb") as tf:
        cfg = tomli.load(tf)

    config = DPLConfig(**cfg)
    level = logging.INFO
    queue = None
    listener = None
    logger_name = __package__  # only set when this module is imported

    if config.num_procs == 1:
        logger = setup_sp_logger(filename=args.log_filename, name=logger_name, level=level)
    else:
        manager = mp.Manager()
        queue = manager.Queue(-1)

        listener = mp.Process(
            target=logging_listener, args=(queue, args.log_filename, logger_name, level)
        )
        listener.start()

        logger = logging.getLogger(logger_name)
        setup_mp_logger(logger, level, queue)

    logger.info(f"Started preprocessor on {datetime.datetime.now().strftime('%Y-%m-%d')}")

    config.filter.elements = list(elements.BASE_ELEMENTS | set(config.filter.elements))

    config_dump = json.dumps(config.model_dump(), indent=2)
    logger.info(f"Processing as per configuration:\n{config_dump}")

    transform_from_file_name = ""

    if config.transform_file:
        transform_filename = pathlib.Path(config.transform_file).resolve()
        logger.info(f"Reading transforms from {transform_filename}")

        with open(transform_filename, "r") as tfile:
            lines = tfile.readlines()

        if lines[0].startswith("// TRANSFORM_NAME:"):
            transform_from_file_name = lines[0].replace("// TRANSFORM_NAME:", "").strip()
        else:
            transform_from_file_name = "from_file"

        transform_from_file = "".join(lines)

    transforms = []

    for transform_name in config.filter.transforms:
        if transform_name == transform_from_file_name or transform_name == "from_file":
            transform = transform_from_file
        else:  # built-in transforms
            transform = getattr(normalizer, transform_name, None)

            if not transform:
                msg = f"Unknown transform {transform_name}"
                logger.critical(msg)
                raise RuntimeError(msg)

        transforms.append(transform)

    all_transforms = "".join(transforms)
    logger.info(f"Applied transforms:\n{all_transforms}")

    infile = pathlib.Path(config.input_csv_file).resolve()
    exts = infile.suffixes

    num_procs = min(os.cpu_count(), config.num_procs)

    if num_procs < config.num_procs:
        logger.warning(f"Adjusting number of cores to {num_procs}")
    else:
        logger.info(f"Using {num_procs} {'cores' if num_procs > 1 else 'core'}")

    sep = config.separator

    if exts == [".smi"] or exts == [".smi", ".gz"]:
        if config.smiles_column == "pubchem":
            df = pl.read_csv(
                str(infile),
                separator=sep,
                has_header=False,
                new_columns=["index", config.smiles_column],
            )
        else:
            df = pl.read_csv(
                str(infile), separator=sep, has_header=False, new_columns=[config.smiles_column]
            )
    else:
        df = pl.read_csv(
            str(infile), separator=sep, has_header=True, columns=[config.smiles_column]
        )

    outfile = pathlib.Path(config.output_smiles_file).resolve()

    input_smilies = df[config.smiles_column]
    n_input_smilies = len(input_smilies)

    logger.info(f"Processing {n_input_smilies} input SMILES")

    regex_filter = RegexFilter(config.filter)
    pbar = dict(
        bar_format="Regex:{desc}|{bar}|{elapsed}",
        ascii=True,
        colour="blue",
    )

    chunk_size = config.chunk_size

    if num_procs > 1 and n_input_smilies > 2_000_000:  # arbitrary...
        with mp.Pool(num_procs) as pool:
            regex_smilies = set(
                tqdm(
                    pool.imap(regex_filter, input_smilies, chunksize=chunk_size),
                    total=len(input_smilies),
                    **pbar,
                )
            )

        regex_smilies.discard(None)
    else:
        regex_smilies = set()  # may be an issue with memory for large datasets

        for smiles in tqdm(input_smilies, **pbar):
            if regex_smiles := regex_filter(smiles):
                regex_smilies.add(regex_smiles)

    with open("regex.smi", "w") as sf:
        for smiles in regex_smilies:
            sf.write(f"{smiles}\n")

    # FIXME: won't work in multiprocessing
    if num_procs == 1:
        logger.info(f"Total number of tokens {regex_filter.token_count:_}")
        discarded_tokens = dict(sorted(regex_filter.discarded_tokens.items()))
        logger.info(f"Discarded tokens:\n{discarded_tokens}")

    logger.info(f"{len(regex_smilies)} SMILES after regex filtering")

    pbar.update(bar_format="Chem:{desc}|{bar}|{elapsed}")

    if num_procs > 1:
        rdkit_filter = RDKitFilter(config.filter, all_transforms, level, queue)

        with mp.Pool(num_procs) as pool:
            rdkit_smilies = set(
                tqdm(
                    pool.imap(rdkit_filter, regex_smilies, chunksize=chunk_size),
                    total=len(regex_smilies),
                    **pbar,
                )
            )

        rdkit_smilies.discard(None)
    else:
        rdkit_filter = RDKitFilter(config.filter, all_transforms)
        rdkit_smilies = set()

        for smiles in tqdm(regex_smilies, **pbar):
            if rdkit_smiles := rdkit_filter(smiles):
                rdkit_smilies.add(rdkit_smiles)

    logger.info(f"{len(rdkit_smilies)} SMILES after chemistry filtering")
    logger.info(f"Writing filtered SMILES to {outfile}")

    with open(config.output_smiles_file, "w") as sf:
        for smiles in rdkit_smilies:
            sf.write(f"{smiles}\n")

    found_tokens = Counter()

    for smiles in rdkit_smilies:
        if smiles:  # FIXME: why are the None in rdkit_smilies?
            tokens = SMILES_TOKENS_REGEX.findall(smiles)
            found_tokens.update(tokens)

    found_tokens = dict(sorted(found_tokens.items()))
    logger.info(f"Final supported tokens:\n{found_tokens}")

    logger.info(f"Finished preprocessor on {datetime.datetime.now().strftime('%Y-%m-%d')}")

    if queue:
        queue.put_nowait(None)
        listener.join()


def main_script():
    mp.set_start_method("spawn", force=True)

    args = parse_command_line()
    main(args)


if __name__ == "__main__":
    main_script()
