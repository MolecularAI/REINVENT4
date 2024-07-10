"""Create a Reinvent model from a list of SMILES strings"""

import sys
from dataclasses import asdict
import time
import datetime
import uuid
import pprint
import logging

import tomli


try:
    import torchinfo
except ImportError:
    torchinfo = None

from reinvent import version
from reinvent.models import meta_data
from reinvent.models.reinvent.models.model import Model
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer, create_vocabulary
from reinvent.chemistry.file_reader import FileReader

logger = logging.getLogger(__name__)


def create_model(
    num_layers: int,
    layer_size: int,
    dropout: float,
    max_sequence_length: int,
    cell_type: str,
    embedding_layer_size: int,
    layer_normalization: bool,
    standardize: bool,
    input_smiles_path: str,
    output_model_path: str,
    metadata: dict,
) -> Model:
    """Create a Reinvent model from scratch

    Learn the vocabulary from SMILES.  SMILES strings are a complete
    molecule.

    :param num_layers: number of network layers
    :param layer_size: length of each layer
    :param dropout: dropout probability
    :param max_sequence_length: maximum length of the SMILES token sequence for
                                sampling
    :param cell_type: either 'lstm' or 'gru'
    :param embedding_layer_size: length of the embedding layer
    :param layer_normalization: whether to carry out layer normalization
    :param standardize: whether SMILES standardization should be done
    :param input_smiles_path: filename of the input SMILES file
    :param output_model_path: filename to store the Torch pickle file
    :param metadata: metadata for the model file
    :returns: a new Reinvent model
    """

    network_params = {
        "dropout": dropout,
        "layer_size": layer_size,
        "num_layers": num_layers,
        "cell_type": cell_type,
        "embedding_layer_size": embedding_layer_size,
        "layer_normalization": layer_normalization,
    }

    pp_params = pprint.pformat(network_params, indent=4)
    logger.info(f"Setting hyper-parameters:\n{pp_params}")

    metadata = meta_data.ModelMetaData(
        creation_date=time.time(),
        hash_id=None,
        hash_id_format="",
        model_id=uuid.uuid4(),
        origina_data_source=metadata.get("data_source", "unknown"),
        comments=[metadata.get("comment", "")],
    )

    pp_params = pprint.pformat(asdict(metadata), indent=4)
    logger.info(f"Writing metadata to model file:\n{pp_params}")

    logger.info(f"Reading SMILES from {input_smiles_path}")
    reader = FileReader([], None)
    smilies = reader.read_delimited_file(input_smiles_path, standardize=standardize)

    tokenizer = SMILESTokenizer()
    vocabulary = create_vocabulary(smilies, tokenizer=tokenizer)

    model = Model(
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        meta_data=metadata,
        network_params=network_params,
        max_sequence_length=max_sequence_length,
    )

    model.save(output_model_path)

    tokens = model.vocabulary.tokens()
    logger.info(f"Working with {len(tokens)} tokens: {', '.join(tokens)}")
    logger.info(f"Model layout:\n{model.network}")

    if torchinfo:
        summary = torchinfo.summary(model.network, verbose=0)
        logger.info(f"Model summary:\n{summary}")

    return model


def get_config(filename):
    """Get the config from a TOML file"""

    with open(filename, "rb") as tf:
        config = tomli.load(tf)

    return config


def main():

    logging.basicConfig(
        format="%(asctime)s <%(levelname)-4.4s> %(message)s", datefmt="%H:%M:%S", level=logging.INFO
    )

    config = get_config(sys.argv[1])

    files = config["io"]
    network_params = config["network"]
    metadata = config["metadata"]

    input_smiles_path = files["smiles_file"]
    output_model_path = files.get("model_file", "empty.model")

    logger.info(
        f"Creating empty Reinvent prior on {datetime.datetime.now().strftime('%Y-%m-%d')}, "
        f"{version.__copyright__}"
    )

    create_model(
        num_layers=network_params.get("num_layers", 3),
        layer_size=network_params.get("layer_size", 512),
        dropout=network_params.get("dropout", 0.0),
        max_sequence_length=network_params.get("max_sequence_length", 256),
        cell_type=network_params.get("cell_type", "lstm"),
        embedding_layer_size=network_params.get("embedding_layer_size", 256),
        layer_normalization=network_params.get("layer_normalization", False),
        standardize=network_params.get("standardize", True),
        input_smiles_path=input_smiles_path,
        output_model_path=output_model_path,
        metadata=metadata,
    )

    logger.info(f"Finished on {datetime.datetime.now().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
