"""REINVENT4

Namespace setup and backward compatibility
"""

import sys

from reinvent import models
from reinvent.version import *
from reinvent.models.libinvent.models import vocabulary  # not sure why needed


# support compatibility with older model files
PATH_MAP = (
    # R3
    ("reinvent_models.reinvent_core", models.reinvent),
    ("reinvent_models.reinvent_core.models", models.reinvent.models),
    ("reinvent_models.reinvent_core.models.vocabulary", models.reinvent.models.vocabulary),
    ("reinvent_models.lib_invent", models.libinvent),
    ("reinvent_models.lib_invent.models", models.libinvent.models),
    ("reinvent_models.lib_invent.models.vocabulary", models.libinvent.models.vocabulary),
    ("reinvent_models.link_invent", models.linkinvent),
    ("reinvent_models.link_invent.model_vocabulary", models.linkinvent.model_vocabulary),
    (
        "reinvent_models.link_invent.model_vocabulary.vocabulary",
        models.linkinvent.model_vocabulary.vocabulary,
    ),
    (
        "reinvent_models.link_invent.model_vocabulary.model_vocabulary",
        models.linkinvent.model_vocabulary.model_vocabulary,
    ),
    (
        "reinvent_models.link_invent.model_vocabulary.paired_model_vocabulary",
        models.linkinvent.model_vocabulary.paired_model_vocabulary,
    ),
)

for module, path in PATH_MAP:
    if path is None:
        sys.modules[module] = sys.modules[__name__]
    else:
        sys.modules[module] = path
