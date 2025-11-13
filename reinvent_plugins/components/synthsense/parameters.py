import dataclasses
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

from reinvent_plugins.components.synthsense.endpoints import AnyEndpoint, endpoint_from_dict

from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    number_of_steps: list[Optional[int]] = None
    time_limit_seconds: list[Optional[int]] = None
    reaction_step_coefficient: list[Optional[float]] = None
    stock: list[Optional[dict[str, Any]]] = None
    scorer: list[Optional[dict[str, Any]]] = None
    stock_profile: list[Optional[str]] = None
    reactions_profile: list[Optional[str]] = None
    score_to_extract: list[str] = None  # Endpoint selection.
    reference_route_file: list[str] = None

    # Route popularity endpoint parameters
    popularity_threshold: list[Optional[int]] = None
    penalty_multiplier: list[Optional[float]] = None
    consider_subroutes: list[Optional[bool]] = None
    min_subroute_length: list[Optional[int]] = None
    penalize_subroutes: list[Optional[str]] = None

    # Fill-a-Plate params
    bucket_threshold: list[Optional[int]] = None
    min_steps_for_penalization: list[Optional[int]] = None
    penalization_enabled: list[Optional[bool]] = None


@dataclass
class ComponentLevelParameters:
    number_of_steps: Optional[int] = None
    time_limit_seconds: Optional[int] = None
    stock: Optional[dict[str, Any]] = None
    scorer: Optional[dict[str, Any]] = None
    stock_profile: Optional[str] = None
    reactions_profile: Optional[str] = None


def get_num_endpoints(params: Parameters) -> int:
    """Return number of endpoints.

    This is the number of elements in the longest list in params.
    """
    fields = dataclasses.fields(params)
    values = [getattr(params, f.name) for f in fields]
    lengths = [len(v) if isinstance(v, list) else 0 for v in values]
    num_endpoints = max(lengths) if lengths else 0
    return num_endpoints


def component_level_fields() -> list[str]:
    return [f.name for f in dataclasses.fields(ComponentLevelParameters)]


def first_non_None(xs: list) -> Any:
    """Return the first non-None element in the list."""
    if xs is None:
        return None
    for x in xs:
        if x is not None:
            return x
    return None


def all_same(xs: list) -> bool:
    """Check if all elements in the list are the same ignoring None."""
    # Items can be dicts, so we can't convert list to set.
    if xs is None:  # Handle None case
        return True
    if not xs:  # Handle empty list
        return True
    first = first_non_None(xs)
    return all((item is None) or (item == first) for item in xs)


def get_component_level_params(params: Parameters) -> ComponentLevelParameters:
    """Extract component-level parameters from Reinvent scoring component params.

    This function tries to "reverse-engineer" component-level parameters
    from list-based parameters lists.

    This function checks that all values for component-level parameters are the same,
    and returns them as a single ComponentLevelParameters object.

    If some values are different, it raises ValueError.

    :param params: Reinvent scoring component params
    """

    kwargs = {}

    for field in component_level_fields():
        values = getattr(params, field)
        if values is None:
            continue
        if not all_same(values):
            raise ValueError(f"Different values for {field} in synthsense parameters: {values}.")
        first = first_non_None(values)
        if first is not None:
            kwargs[field] = first

    component_params = ComponentLevelParameters(**kwargs)
    return component_params


def slice(params: dict[str, list[Any]], i: int) -> dict[str, Any]:
    """Slice parameters lists on the given index.

    "Slice" parameters lists on index i.
    Example:
    Input: {
      "a": [1, 2],
      "b": [4, 5]
    }
    Output: Slice 0: {"a": 1, "b": 4}, slice 1: {"a": 2, "b": 5}.

    :param params: Dictionary of parameters with lists as values.
    :param i: Index to slice the lists.
    :returns: Dictionary with sliced values.
    """
    return {k: v[i] for k, v in params.items() if v is not None}


def drop_None(d: dict[str, Any]) -> dict[str, Any]:
    """Drop k-v pairs where v is None."""
    return {k: v for k, v in d.items() if v is not None}


def endpoint_from_slice(params: dict[str, list[Any]]) -> AnyEndpoint:
    """Create an endpoint from a slice of parameters."""
    return endpoint_from_dict(drop_None(params))


def get_endpoint_params(params: Parameters) -> list[AnyEndpoint]:
    """Extract endpoint-level parameters from Reinvent scoring component params."""

    complevel_fields = component_level_fields()

    num_endpoints = get_num_endpoints(params)

    # Drop component-level keys, keep only endpoint-level keys.
    ep_params = {k: v for k, v in asdict(params).items() if k not in complevel_fields}

    # "Slice" remaining (endpoint-level) parameters, and create an endpoint from each slice.
    endpoints = [endpoint_from_slice(slice(ep_params, i)) for i in range(num_endpoints)]

    return endpoints


def split_params(params: Parameters) -> Tuple[ComponentLevelParameters, list[AnyEndpoint]]:
    """Splits parameters from lists into individual endpoints.

    Some parameters are component-level parameters,
    and will be lifted from lists into one ComponentLevelParameters.

    Other parameters are endpoint-level parameters,
    and will be split into a list of Endpoint objects.
    """

    component_level_params = get_component_level_params(params)

    endpoints = get_endpoint_params(params)

    return component_level_params, endpoints
