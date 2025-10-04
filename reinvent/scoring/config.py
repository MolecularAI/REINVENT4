"""Convert the components dictionary into an internal data structure"""

__all__ = ["get_components"]
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
import logging

from .importer import get_registry
from .transforms.transform import get_transform

import pumas
from pumas.desirability import desirability_catalogue

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


def get_components(components: list[dict[str, dict]], use_pumas: bool = False) -> ComponentType:
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
        component_type, component_value = list(component.items())[0]
        endpoints: dict = component_value["endpoint"]
        complevel_params = component_value.get("params", {})  # Component-level params, if exist.

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

            # Merge component-level params with this endpoint params.
            # Endpoint params take precedence.
            merged_params = {**complevel_params, **params}

            parameters.append(merged_params)

            transform = None

            if transform_params:
                if use_pumas:
                    transform_type = transform_params['type'].lower()
                else:
                    transform_type = transform_params["type"].lower().replace("-", "").replace("_", "")

                try:
                    if use_pumas:
                        Transform = desirability_catalogue.get(transform_type)
                    else:
                        Transform, TransformParams = get_transform(transform_type)
                except AttributeError:
                    raise RuntimeError(f"Unknown transform type: {transform_params['type']}")
                except Exception as e:
                    raise Exception(f'An unhandled exception occured: {e}')

                if use_pumas:
                    # Gather the expected parameters for the transform and their types
                    param_defs = Transform().parameter_manager.parameter_definitions
                    params = {}

                    # Convert the parameters to the expected types
                    # TODO: reimplement type checking and conversion in pumas with type coercion
                    for key, value in transform_params.items():
                        if key == 'type':
                            continue
                        elif value is not None and key in param_defs and param_defs[key]['type'] == 'float':
                            params[key] = float(value)
                        elif value is not None and key in param_defs and param_defs[key]['type'] == 'int':
                            params[key] = int(value)
                        else:
                            params[key] = value
                    transform = Transform(params=params)
                else:
                    transform = Transform(TransformParams(**transform_params))

            transforms.append(transform)
            weights.append(weight)
            names.append(name)

        component_params = None
        collected_params = collect_params(parameters)

        if ComponentParams:
            component_params = ComponentParams(**collected_params)

        component = Component(component_params)
        data = ComponentData(
            component_type=component_type_lookup,
            params=(names, component, transforms, weights),
            cache=defaultdict(dict),
        )

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


def collect_params(params: List[Dict]) -> Dict[str, List[Any]]:
    """Convert a list of dictionaries to a dictionary

    Collect the values with the same key in each dictionary of the list into
    a dictionary. The number of key/value pairs in the dictionaries in each
    item of the passed in parameters may be different. Missing keys will be
    filled with None.

    :param params: list of dictionaries
    :returns: a dictionary
    """

    collected_params = defaultdict(list)

    # Collect all keys from all dictionaries
    keys = set()
    for param_dict in params:
        keys.update(param_dict.keys())

    # Go through each dictionary and collect values for each key
    for param_dict in params:
        for key in keys:
            collected_params[key].append(param_dict.get(key, None))
    return collected_params
