"""Compute the scores and the transformed scores"""

__all__ = ["compute_transform"]

from typing import List, Tuple, Callable, Optional
import logging

import numpy as np

from reinvent_plugins.components.component_results import ComponentResults
from .results import TransformResults


logger = logging.getLogger(__name__)
SCORE_FUNC = Callable[[List[str]], ComponentResults]


def compute_component_scores(
    smilies: List[str],
    scoring_function: SCORE_FUNC,
    cache,
    filter_mask: Optional[np.ndarray[bool]],
) -> ComponentResults:
    """Compute a single component's scores and cache the results

    The mask filters out all SMILES unwanted for score computation: SMILES not
    passing a previous filter, invalid SMILES, duplicate SMILES.  Scores are NaN
    when their values are unknown.

    :param smilies: list of SMILES
    :param scoring_function: the component callable
    :param cache: the cache for the component (will be modified)
    :param filter_mask: array mask to filter out invalid and duplicate SMILES
    :returns: the scores
    """

    # NOTE: if a component has multiple endpoints it needs to declare this!
    number_of_endpoints = getattr(scoring_function, "number_of_endpoints", 1)

    masked_scores = [(np.nan,)] * len(smilies)

    for i, value in enumerate(filter_mask):
        if not value:  # the SMILES will not be passed to scoring_function
            masked_scores[i] = (0.0,) * number_of_endpoints

    scores = {}  # keep track of scores for each SMILES as there may be duplicates

    for smiles, score in zip(smilies, masked_scores):
        if smiles not in scores:  # do not overwrite duplicates with zeroes
            scores[smiles] = score

    smilies_non_cached = []

    for smiles, score in scores.items():
        if smiles in cache:
            scores[smiles] = cache[smiles]
        elif any(score):  # filter out SMILES already scored zero, will catch np.nan
            smilies_non_cached.append(smiles)
        # If score is zero we do not need to compute the score

    if smilies_non_cached:
        component_results = scoring_function(smilies_non_cached)
    else:  # only duplicates or masked: need to set noe "empty" ComponentResults
        _scores = [[] for _ in range(number_of_endpoints)]
        component_results = ComponentResults(_scores)

    for data in zip(smilies_non_cached, *component_results.scores):
        smiles = data[0]
        component_scores = data[1:]

        scores[smiles] = component_scores

    cache.update(((k, v) for k, v in scores.items()))

    # add cached scores to ComponentResults
    scores_values = [scores[smiles] for smiles in smilies]  # expand scores
    component_results.scores = [np.array(arr) for arr in zip(*scores_values)]

    return component_results


def compute_transform(
    component_type,
    params: Tuple,
    smilies: List[str],
    caches: dict,
    valid_mask: np.ndarray[bool],
) -> TransformResults:
    """Compute the component score and transform of it

    :param component_type: type of the component
    :param params: parameters for the component
    :param smilies: list of SMILES
    :param caches: the component's cache
    :param valid_mask: mask for valid SMILES, i.e. false for invalid
    :returns: dataclass with transformed results
    """

    names, scoring_function, transforms, weights = params

    component_results = compute_component_scores(
        smilies, scoring_function, caches[component_type], valid_mask
    )

    transformed_scores = []

    for scores, transform in zip(component_results.scores, transforms):
        transformed = transform(scores) if transform else scores
        transformed_scores.append(transformed * valid_mask)

    transform_types = [transform.params.type if transform else None for transform in transforms]

    transform_result = TransformResults(
        component_type,
        names,
        transform_types,
        transformed_scores,
        component_results,
        weights,
    )

    return transform_result
