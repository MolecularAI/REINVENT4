import importlib
import importlib.resources
from contextlib import contextmanager

import reinvent.chemistry.library_design.reaction_definitions.data


@contextmanager
def default_reaction_definitions():
    with importlib.resources.path(
        reinvent.chemistry.library_design.reaction_definitions.data, "reaction_definitions.csv"
    ) as reaction_definitions_path:
        yield reaction_definitions_path
