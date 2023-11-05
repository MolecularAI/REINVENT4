"""Create a Reinvent model from a list of SMILES strings"""

from __future__ import annotations

from reinvent.models.reinvent.models.model import Model
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer, create_vocabulary
from reinvent.chemistry.file_reader import FileReader


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
    :returns: a new Reinvent model
    """

    reader = FileReader([], None)
    smiles_list = reader.read_delimited_file(input_smiles_path, standardize=standardize)

    tokenizer = SMILESTokenizer()
    vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)

    network_params = {
        "dropout": dropout,
        "layer_size": layer_size,
        "num_layers": num_layers,
        "cell_type": cell_type,
        "embedding_layer_size": embedding_layer_size,
        "layer_normalization": layer_normalization,
    }

    model = Model(
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        network_params=network_params,
        max_sequence_length=max_sequence_length,
    )

    model.save(output_model_path)

    return model


if __name__ == "__main__":
    import sys

    input_smiles_path = sys.argv[1]
    output_model_path = sys.argv[2]

    create_model(
        num_layers=3,
        layer_size=512,
        dropout=0.0,
        max_sequence_length=256,
        cell_type="lstm",
        embedding_layer_size=256,
        layer_normalization=False,
        standardize=False,
        input_smiles_path=input_smiles_path,
        output_model_path=output_model_path,
    )
