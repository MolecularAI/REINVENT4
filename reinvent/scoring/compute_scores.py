"""Compute the scores and the transformed scores"""

__all__ = ["compute_transform"]

from typing import List, Tuple, Callable, Optional
import logging

import numpy as np

from reinvent_plugins.components.component_results import (
    ComponentResults,
    SmilesAssociatedComponentResults,
)
from .results import TransformResults


logger = logging.getLogger(__name__)
SCORE_FUNC = Callable[[List[str]], ComponentResults]


def compute_component_scores(
    smilies: List[str],
    scoring_function: SCORE_FUNC,
    cache,
    filter_mask: Optional[np.ndarray[bool]],
    index_smiles: Optional[List[str]] = None,
) -> SmilesAssociatedComponentResults:
    """Compute a single component's scores and cache the results

    The mask filters out all SMILES unwanted for score computation: SMILES not
    passing a previous filter, invalid SMILES.  Scores are NaN
    when their values are unknown.
    Note: duplicates need to be handled above this level since ComponentResults is SMILES based

    The logic is as follows:
     1 get the smiles that need to be scored in this call
       this is all inputs except for masked smiles or those in cache
     2 compute the score for the new smiles and add to cache, or create a blank ComponentResults object
     3 update the scores of cached smiles to the SmilesResult object (scores + metadata) for cache hits
     4 add zeros-scores for any unique masked SMILES, e.g. invalid but non-duplicated

     In the case of fragments, we get SMILES from the fragment not the full molecule, but for consistency
     in output writing, caching scores etc

    :param smilies: list of SMILES
    :param scoring_function: the component callable
    :param cache: the cache for the component (will be modified)
    :param filter_mask: array mask to filter out invalid SMILES
    :param index_smiles: list of SMILES to index scores, used when the scored SMILES are fragments
    :returns: the scores
    """
    smilies_to_score, smiles_with_masked_scores, cache_hits = [], [], []

    for filter_flag, smiles in zip(filter_mask, smilies):
        if filter_flag:
            if smiles in cache.keys():
                cache_hits.append(smiles)
            else:
                smilies_to_score.append(smiles)
        else:
            smiles_with_masked_scores.append(smiles)

    # we need different behaviour for duplicates and invalids - duplicates should not overwrite the scores
    # in ComponentResults. Therefore, we should always keep the scored version of smilies if it occurs with & wihtout filters
    smiles_with_masked_scores = [
        smiles
        for smiles in smiles_with_masked_scores
        if not smiles in smilies_to_score + cache_hits
    ]

    # handle the case of fragement SMILES, here we will use the full SMILES to keep the score asscociated with the record
    # while the fragement only for score computation
    if index_smiles is not None:
        index_smiles_to_score = [index_smiles[smilies.index(s)] for s in smilies_to_score]
        index_smiles_with_masked_scores = [
            index_smiles[smilies.index(s)] for s in smiles_with_masked_scores
        ]
        cache_hits = [index_smiles[smilies.index(s)] for s in cache_hits]
        logger.debug(
            f"Using index smilies for fragement component {type(scoring_function).__name__}"
        )
    else:
        index_smiles_to_score = smilies_to_score
        index_smiles_with_masked_scores = smiles_with_masked_scores

    # debug statement here as invalids are passed in as "None" instead of smiles.
    logger.debug(
        f"Masked smilies for {type(scoring_function).__name__} are {smiles_with_masked_scores}"
    )

    if len(smilies_to_score) > 0:
        component_results = SmilesAssociatedComponentResults(
            component_results=scoring_function(smilies_to_score), smiles=index_smiles_to_score
        )

        # update cache
        cache.update((smiles, component_results[smiles]) for smiles in index_smiles_to_score)

    else:
        # in this case, there are no compounds to score. Create blank ComponentResults
        component_results = SmilesAssociatedComponentResults.create_from_scores(
            smiles=[], scores=[[]]
        )

    if len(cache_hits) > 0:  # update the results
        for smiles in cache_hits:
            component_results.data[smiles] = cache[smiles]

    if len(smiles_with_masked_scores) > 0:
        # one score per endpoint
        masked_scores = [(0.0,) * len(smiles_with_masked_scores)] * getattr(
            scoring_function, "number_of_endpoints", 1
        )

        component_results.update_scores(
            smiles=index_smiles_with_masked_scores, scores=masked_scores
        )

    return component_results


def compute_transform(
    component_type,
    params: Tuple,
    smilies: List[str],
    caches: dict,
    valid_mask: np.ndarray[bool],
    index_smiles: Optional[List[str]] = None,
    pumas: bool = False,
) -> TransformResults:
    """Compute the component score and transform of it

    :param component_type: type of the component
    :param params: parameters for the component
    :param smilies: list of SMILES
    :param caches: the component's cache
    :param valid_mask: mask for valid SMILES, i.e. false for invalid
    :param index_smiles: list of SMILES to index scores, used when the scored SMILES are fragments
    :returns: dataclass with transformed results
    """

    names, scoring_function, transforms, weights = params

    component_results = compute_component_scores(
        smilies, scoring_function, caches[component_type], valid_mask, index_smiles
    )

    transformed_scores = []
    # this loop is over multiple scores per component
    for scores, transform in zip(
        component_results.fetch_scores(
            smiles=index_smiles if index_smiles is not None else smilies, transpose=True
        ),
        transforms,
    ):
        if pumas:
            vectorise_transform = np.vectorize(transform)
            transformed = vectorise_transform(scores) if transform else scores
            transformed_scores.append(transformed * valid_mask)
        else:
            transformed = transform(scores) if transform else scores
            transformed_scores.append(transformed * valid_mask)

    if pumas:
        transform_types= [ None for transform in transforms] #TODO This needs fixing on the pumas end. There is no way to fetch the type from the transfom object
    else:
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
