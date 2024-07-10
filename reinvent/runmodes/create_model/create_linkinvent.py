"""Create a Linkinvent model from a list of SMILES strings"""

from __future__ import annotations
import time
import uuid

from reinvent.models.linkinvent.link_invent_model import LinkInventModel
from reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary import (
    PairedModelVocabulary,
)
from reinvent.models.linkinvent.networks import EncoderDecoder
from reinvent.chemistry.file_reader import FileReader
from reinvent.models import meta_data


def create_model(
    num_layers: int,
    layer_size: int,
    dropout: float,
    max_sequence_length: int,
    input_smiles_path: str,
    output_model_path: str,
):
    """Create a Linkinvent model from scratch

    Learn the vocabulary from SMILES.  Note that the warheads ('input') must
    come in column 1 and the linker ('target' or 'output') in column 2.

    :param num_layers: number of network layers
    :param layer_size: length of each layer
    :param dropout: dropout probability
    :param max_sequence_length: maximum length of the SMILES token sequence for
                                sampling
    :param input_smiles_path: filename of the input SMILES file with warheads
                              and linkers (see above)
    :param output_model_path: filename to store the Torch pickle file
    :returns: a new Linkinvent model
    """

    reader = FileReader([], None)

    # build vocabulary
    warheads, linker = zip(*reader.read_library_design_data_file(input_smiles_path, num_fields=2))

    vocabulary = PairedModelVocabulary.from_lists(warheads, linker)

    encoder_config = {
        "num_layers": num_layers,
        "num_dimensions": layer_size,
        "dropout": dropout,
        "vocabulary_size": len(vocabulary.input),  # FIXME: check if this is sane
    }

    decoder_config = {**encoder_config, "vocabulary_size": len(vocabulary.target)}

    network = EncoderDecoder(encoder_config, decoder_config)

    metadata = meta_data.ModelMetaData(
        creation_date=time.time(),
        hash_id=None,
        hash_id_format="",
        model_id=uuid.uuid4(),
        origina_data_source="unknown",
        comments=[],
    )

    model = LinkInventModel(
        vocabulary=vocabulary,
        network=network,
        meta_data=metadata,
        max_sequence_length=max_sequence_length,
    )

    model.save_to_file(output_model_path)  # torch.save()

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
        input_smiles_path=input_smiles_path,
        output_model_path=output_model_path,
    )
