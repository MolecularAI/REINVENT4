"""Evaluate functions for TB"""

from __future__ import annotations

__all__ = [
    "mutual_similarities",
    "internal_diversity",
    "compute_similarity_from_sample",
]
from typing import List, Iterable

from rdkit import DataStructs
import numpy as np


def mutual_similarities(fps: Iterable[float]) -> np.ndarray:
    """Mutual similarities among a set of fingerprints

    :param fps: fingerprints
    :returns: 1D numpy vector representing a triangle matrix
    """

    nfps = len(fps)
    similarity = []

    for n in range(nfps - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n + 1 :])
        similarity.extend([s for s in sim])

    return np.array(similarity)


def internal_diversity(similarities: np.ndarray, p: int = 1) -> float:
    """Internal diversity, see https://www.frontiersin.org/articles/10.3389/fphar.2020.565644/

    We assume that similarities is a triangle matrix.
    FIXME: look into iSIM to see if that can be used here

    :param similarities: 1D similarity array constructed from the triangle(!) matrix
    :param p: order
    :returns: internal diversity
    """

    size = len(similarities)
    g = size * (size - 1) / 2
    s = sum(similarities**p) / g

    return 1.0 - np.power(s, 1.0 / p)


def compute_similarity_from_sample(fps: Iterable, ref_fps: Iterable):
    """Take the first SMIlES from the input set and compute the
    average similarity from SMILES from a sample

    :param fps: list of SMILES
    :param ref_fps: reference fingerprints
    """

    sims = []

    for ref_fp in ref_fps:
        sims.append(np.array(DataStructs.BulkTanimotoSimilarity(ref_fp, fps)))

    similarities = np.array(sims).mean(axis=0)

    return similarities
