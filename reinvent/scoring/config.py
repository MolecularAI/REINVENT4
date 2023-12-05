"""Convert the components dictionary into an internal data structure"""

__all__ = ["get_components"]
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import logging

from .importer import get_registry
from .transforms.transform import get_transform


logger = logging.getLogger(__name__)


@dataclass
class ComponentType:
    scorers: List
    filters: List
    penalties: List


@dataclass
class ComponentData:
    component_type: str
    params: Tuple
    cache: Dict


def get_components(components: dict[str, dict]) -> ComponentType:
    """Get all the components from the configuration

    Stores the component function, transform and results objects.

    :param components: the component configurations
    :returns: a dict with the objects
    """
    # these need to be lists to store copies with the same name
    scorers = []
    filters = []
    penalties = []

    component_registry = get_registry()

    for component in components:
        component_type, component_value  = list(component.items())[0]
        endpoints: dict = component_value["endpoint"]

        component_type_lookup = component_type.lower().replace("-", "").replace("_", "")

        try:
            Component, ComponentParams = component_registry[component_type_lookup]
        except AttributeError:
            raise RuntimeError(f"Unknown scoring component: {component_type}")

        parameters = []
        transforms = []
        weights = []
        names = []

        for endpoint in endpoints:
            name: str = endpoint.get("name", component_type)
            params: dict = endpoint.get("params", {})
            transform_params: Optional[dict] = endpoint.get("transform", None)
            weight: float = endpoint.get("weight", 1.0)

            if weight < 0:
                raise RuntimeError(f"weight must be equal to or larger than zero but is {weight}")

            parameters.append(params)

            transform = None

            if transform_params:
                transform_type = transform_params["type"].lower().replace("-", "").replace("_", "")

                try:
                    Transform, TransformParams = get_transform(transform_type)
                except AttributeError:
                    raise RuntimeError(f"Unknown transform type: {transform_params['type']}")

                transform = Transform(TransformParams(**transform_params))

            transforms.append(transform)
            weights.append(weight)
            names.append(name)

        component_params = None
        collected_params = collect_params(parameters)

        if ComponentParams:
            component_params = ComponentParams(**collected_params)

        component = Component(component_params)
        data = ComponentData(component_type = component_type_lookup,
                             params = (names,component,transforms,weights),
                             cache = defaultdict(dict))


        if Component.__component == "filter":
            logger.info(f"Creating filter component {component_type}")
            filters.append(data)
        elif Component.__component == "penalty":
            logger.info(f"Creating penalty component {component_type}")
            penalties.append(data)
        else:
            logger.info(f"Creating scoring component {component_type}")
            scorers.append(data)

    return ComponentType(scorers, filters, penalties)


def collect_params(params: List[Dict]) -> defaultdict:
    """Convert a list of dictionaries to a dictionary

    Collect the values with the same key in each dictionary of the list into
    a dictionary. The number of key/value pairs in the dictionaries in each
    item of the passed in parameters may be different.

    :param params: list of dictionaries
    :returns: a dictionary
    """

    collected_params = defaultdict(list)

    for param_dict in params:
        for key, value in param_dict.items():
            collected_params[key].append(value)

    return collected_params
