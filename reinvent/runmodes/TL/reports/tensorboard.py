"""Write out a TensorBoard report"""

from __future__ import annotations
from typing import List, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs

from reinvent.runmodes.utils import make_grid_image, compute_similarity_from_sample


ROWS = 5
COLS = 6


@dataclass
class TBData:
    epoch: int
    mean_nll: float
    sampled_smilies: Sequence
    sampled_nlls: Sequence
    fingerprints: Sequence
    reference_fingerprints: Sequence
    fraction_valid: float
    number_duplicates: float
    internal_diversity: float
    mean_nll_validation: float = None


def write_report(reporter, data) -> None:
    """Write out TensorBoard data

    :param reporter: TB reporter for writing out the data
    :param data: data to be written out
    """

    mean_nll_stats = {
        "Training Loss": data.mean_nll,
        "Sample Loss": data.sampled_nlls.mean(),
    }

    if data.mean_nll_validation is not None:
        mean_nll_stats["Validation Loss"] = data.mean_nll_validation

    reporter.add_scalars("A_Mean NLL loss", mean_nll_stats, data.epoch)

    reporter.add_scalar("B_Fraction valid SMILES", data.fraction_valid, data.epoch)
    reporter.add_scalar("C_Duplicate SMILES (per epoch)", data.number_duplicates, data.epoch)

    if data.internal_diversity > 0.0:
        reporter.add_scalar(
            "D_Internal Diversity of sample", data.internal_diversity, data.epoch
        )

    # FIXME: rows and cols depend on sample_batch_size
    image_tensor, nimage = make_grid_image(
        data.sampled_smilies, data.sampled_nlls, "NLL", ROWS * COLS, ROWS
    )

    if image_tensor is not None:
        reporter.add_image(
            f"Sampled structures",
            image_tensor,
            data.epoch,
            dataformats="CHW",
        )  # channel, height, width

    if data.reference_fingerprints:
        similarities = compute_similarity_from_sample(
            data.fingerprints, data.reference_fingerprints
        )
        reporter.add_histogram(
            "Tanimoto similarity on RDKitFingerprint", similarities, data.epoch
        )
