"""Compute weighted means"""

__all__ = ["arithmetic_mean", "geometric_mean", "custom_product", "custom_sum"]
from typing import List, Literal, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


def _aggregate(
    all_scores: List[Tuple[np.ndarray, float]], mode: Literal["sum", "prod"]
) -> np.ndarray:
    """Compute a weighted aggregated score

    The weights will be normalized.

    :param all_scores: a list of scores and weights
    :return: aggregated scores
    """

    sizes = {len(scores) for scores, _ in all_scores}
    if len(sizes) > 1:
        raise ValueError(f"Mismatch in number of scores, got {sizes}")

    scores = np.array([score for score, _ in all_scores])
    weights = np.array([weight for _, weight in all_scores])

    nans = np.isnan(scores)
    if nans.any():
        logger.debug("NaN in component score")

    # We broadcast the 1D weights (n_score_components) to the 2D score array shape
    # (n_score_components, n_smiles) here to be able to mask some values as NaN to
    # avoid incurring a faulty weight normalization. We also need an array copy
    # here because broadcast returns a view by default.
    weights = np.array(np.broadcast_to(weights.reshape(-1, 1), scores.shape), dtype=np.float32)
    weights[nans] = np.nan

    if mode == "sum":
        sum_weights = np.nansum(weights, axis=0)
        result = np.nansum(scores * (weights / sum_weights), axis=0)
    elif mode == "prod":
        scores = np.maximum(scores, 1e-8)
        sum_weights = np.maximum(np.nansum(weights, axis=0), 1e-8)
        result = np.nanprod(scores ** (weights / sum_weights), axis=0)
    else:
        raise ValueError(f"Invalid mode '{mode}'")
    return result


def arithmetic_mean(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the weighted arithmetic mean

    The weights will be normalized.

    :param all_scores: a list of scores and weights
    :return: aggregated scores
    """

    return _aggregate(all_scores, mode="sum")


custom_sum = arithmetic_mean


def geometric_mean(all_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Compute the weighted geometric mean

    The weights will be normalized.

    :param all_scores: a list of scores and weights
    :return: aggregated scores
    """
    return _aggregate(all_scores, mode="prod")


custom_product = geometric_mean
