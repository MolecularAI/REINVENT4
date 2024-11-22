"""A simple plugin system for scoring components"""

__all__ = ["get_registry"]
import importlib
import pkgutil
import dataclasses
from typing import Tuple
import logging

from reinvent_plugins import components  # FIXME: this is not the solution

if components.__file__ is not None:  # this is not a namespace package
    raise RuntimeError("No valid component directories found")


logger = logging.getLogger(__name__)


def iter_namespace(ns_pkg):
    return pkgutil.walk_packages(ns_pkg.__path__, ns_pkg.__name__ + ".")


def get_registry() -> dict[str, Tuple[type, type]]:
    registry = {}

    for _, name, ispkg in iter_namespace(components):
        if ispkg:
            continue

        basename = name.split(".")[-1]

        if not basename.startswith("comp_"):
            continue

        try:
            module = importlib.import_module(name)
        except ImportError as e:
            logger.error(f"Component {name} could not be imported: {e}")
            continue

        component_classes = []
        param_class = None

        for attr in module.__dict__.values():
            if isinstance(attr, type):
                if "__component" in attr.__dict__:
                    component_classes.append(attr)

                if "__parameters" in attr.__dict__ and dataclasses.is_dataclass(attr):
                    param_class = attr

        for component in component_classes:
            cls_name = component.__name__.lower()  # canonicalize name
            registry[cls_name] = (component, param_class)

            logger.info(f"Registered scoring component {component.__name__}")

    return registry
