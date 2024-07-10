"""Create a Mol2Mol model from a list of SMILES strings"""

from __future__ import annotations
import time
import uuid

import csv

from reinvent.models import Mol2MolModel, meta_data
from reinvent.models.transformer.transformer import EncoderDecoder
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer


def create_model(
    num_layers: int,
    dropout: float,
    max_sequence_length: int,
    num_heads: int,
    model_dimension: int,
    feedforward_dimension: int,
    input_smiles_path: str,
    output_model_path: str,
):
    """Create a Mol2Mol model from scratch

    Learn the vocabulary from SMILES.

    :param num_layers: number of network layers
    :param dropout: dropout probability
    :param max_sequence_length: maximum length of the SMILES token sequence for
                                sampling
    :param num_heads: number of attention heads
    :param model_dimension: model dimension(?)
    :param feedforward_dimension:
    :param input_smiles_path: filename of the input SMILES file
    :param output_model_path: filename to store the Torch pickle file
    :returns: a new Mol2Mol model
    """

    vocabulary = build_vocabulary(input_smiles_path)

    config = {
        "vocabulary_size": len(vocabulary),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "model_dimension": model_dimension,
        "feedforward_dimension": feedforward_dimension,
        "dropout": dropout,
    }

    network = EncoderDecoder(**config)

    metadata = meta_data.ModelMetaData(
        creation_date=time.time(),
        hash_id=None,
        hash_id_format="",
        model_id=uuid.uuid4(),
        origina_data_source="unknown",
        comments=[],
    )

    model = Mol2MolModel(
        vocabulary=vocabulary,
        network=network,
        meta_data=metadata,
        max_sequence_length=max_sequence_length,
    )

    model.save_to_file(output_model_path)

    return model


def build_vocabulary(input_smiles_path) -> Vocabulary:

    smilies = []

    with open(input_smiles_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")

        for row in reader:
            smilies.append(row[0])

    tokenizer = SMILESTokenizer()
    tokens = set()

    for smiles in smilies:
        tokens.update(tokenizer.tokenize(smiles, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["*", "^", "$"] + sorted(tokens))  # pad=0, start=1, end=2

    # FIXME: where does that come from?
    # for random smiles
    if "8" not in vocabulary.tokens():
        vocabulary.update(["8"])

    return vocabulary


if __name__ == "__main__":
    import sys

    input_smiles_path = sys.argv[1]
    output_model_path = sys.argv[2]

    create_model(
        num_layers=6,
        num_heads=8,
        model_dimension=256,
        feedforward_dimension=2048,
        dropout=0.1,
        max_sequence_length=256,
        input_smiles_path=input_smiles_path,
        output_model_path=output_model_path,
    )
