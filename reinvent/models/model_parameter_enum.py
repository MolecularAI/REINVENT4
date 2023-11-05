from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParametersEnum:
    NUMBER_OF_LAYERS = "num_layers"
    NUMBER_OF_DIMENSIONS = "num_dimensions"
    VOCABULARY_SIZE = "vocabulary_size"
    DROPOUT = "dropout"
