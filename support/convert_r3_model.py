"""A simple script to convert reinvent3 models to reinvent4 models

This script will update the paths to the model classes and add meta information
to the model file.  Run this after having created the symlinks with the script
in support.
"""

import os
from dataclasses import dataclass
from typing import Tuple
import argparse
import sys

import torch

from reinvent.models.mol2mol.models.vocabulary import Vocabulary
import reinvent.models.reinvent
import reinvent.models.libinvent.models.vocabulary
import reinvent.models.mol2mol


@dataclass
class MolformerNetworkParameters:
    vocabulary_size: int = 0
    num_layers: int = 6
    num_heads: int = 8
    model_dimension: int = 256
    feedforward_dimension: int = 2048
    dropout: float = 0.1


@dataclass
class MolformerModelParameterDTO:
    vocabulary: Vocabulary
    max_sequence_length: int
    network_parameter: MolformerNetworkParameters
    network_state: dict


PATH_MAP = (
    ("reinvent_models.reinvent_core", reinvent.models.reinvent),
    ("reinvent_models.reinvent_core.models", reinvent.models.reinvent.models),
    ("reinvent_models.reinvent_core.models.vocabulary", reinvent.models.reinvent.models.vocabulary),
    ("reinvent_models.lib_invent", reinvent.models.libinvent),
    ("reinvent_models.lib_invent.models", reinvent.models.libinvent.models),
    ("reinvent_models.lib_invent.models.vocabulary", reinvent.models.libinvent.models.vocabulary),
    ("reinvent_models.link_invent", reinvent.models.linkinvent),
    ("reinvent_models.link_invent.model_vocabulary", reinvent.models.linkinvent.model_vocabulary),
    (
        "reinvent_models.link_invent.model_vocabulary.vocabulary",
        reinvent.models.linkinvent.model_vocabulary.vocabulary,
    ),
    (
        "reinvent_models.link_invent.model_vocabulary.model_vocabulary",
        reinvent.models.linkinvent.model_vocabulary.model_vocabulary,
    ),
    (
        "reinvent_models.link_invent.model_vocabulary.paired_model_vocabulary",
        reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary,
    ),
    ("reinvent_models.molformer", reinvent.models.mol2mol),
    ("reinvent_models.molformer", reinvent.models.mol2mol),
    ("reinvent_models.molformer.models", reinvent.models.mol2mol.models),
    ("reinvent_models.molformer.models.vocabulary", reinvent.models.mol2mol.models.vocabulary),
    ("reinvent_models.molformer.dto.molformer_model_parameters_dto", None),
)


def convert_model(in_filename: str, out_filename: str) -> None:
    """Convert a model and write it out.

    :param in_filename: REINVENT 3 model file
    :param out_filename: REINVENT 4 model file
    """

    for module, path in PATH_MAP:
        if path is None:
            sys.modules[module] = sys.modules[__name__]
        else:
            sys.modules[module] = path

    model = torch.load(in_filename, weights_only=False)

    if "network_parameter" in model and isinstance(
        model["network_parameter"], MolformerNetworkParameters
    ):
        model["network_parameter"] = model["network_parameter"].__dict__

    if "network" in model:
        model_type = "Reinvent"
    elif "model" in model:
        model_type = "Libinvent"
    elif "encoder_params" in model["network_parameter"]:
        model_type = "Linkinvent"
    elif "num_heads" in model["network_parameter"]:
        model_type = "Mol2Mol"
    else:
        model_type = None

    if model_type:
        model["model_type"] = model_type
        model["version"] = 4
    else:
        raise RuntimeError(f"Model file {in_filename} is of unknown format")

    for module in sys.modules.copy():
        if "reinvent_models" in module:
            del sys.modules[module]

    torch.save(model, out_filename)


def parse_command_line() -> Tuple[str, str]:
    """Parse the command line.

    :return: input and output filename
    """

    parser = argparse.ArgumentParser(
        description=f"A simple script to convert REINVENT 3 models to REINVENT 4 models."
        "Ensure that the relevant symlinks are in place, see support/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "reinvent3_model_file",
        default=None,
        metavar="INFILE",
        type=os.path.abspath,
        help="REINVENT 3 model input file",
    )

    parser.add_argument(
        "reinvent4_model_file",
        default=None,
        metavar="OUTFILE",
        type=os.path.abspath,
        help="REINVENT 4 model output file",
    )

    args = parser.parse_args()

    return args.reinvent3_model_file, args.reinvent4_model_file


if __name__ == "__main__":
    _in_filename, _out_filename = parse_command_line()
    convert_model(_in_filename, _out_filename)
