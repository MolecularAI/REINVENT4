"""Write out a TensorBoard report"""

from __future__ import annotations
from typing import List, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs

from reinvent.runmodes.utils import make_grid_image


ROWS = 5
COLS = 6


@dataclass
class TBData:
    epoch: int
    mean_nll: float
    ref_fps: List
    sampled_smilies: Sequence
    sampled_nlls: Sequence
    fraction_valid: float
    mean_nll_validation: float = None


def write_report(reporter, data, duplicates) -> None:
    """Write out TensorBoard data

    :param reporter: TB reporter for writing out the data
    :param data: data to be written out
    :param duplicates: SMILES cache
    """

    mean_nll_stats = {"Training Loss": data.mean_nll, "Sample Loss": data.sampled_nlls.mean()}

    if data.mean_nll_validation is not None:
        mean_nll_stats["Validation Loss"] = data.mean_nll_validation

    reporter.add_scalars("A_Mean NLL loss", mean_nll_stats, data.epoch)

    reporter.add_scalar("B_Fraction valid SMILES", data.fraction_valid, data.epoch)
    reporter.add_scalar("C_Duplicate SMILES", len(duplicates), data.epoch)

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

    if data.ref_fps:
        similarities = compute_similarity_from_sample(data.sampled_smilies, data.ref_fps)
        reporter.add_histogram("Tanimoto similarity on RDKitFingerprint", similarities, data.epoch)


def compute_similarity_from_sample(smilies: List, ref_fps: List):
    """Take the first SMIlES from the input set and compute ther
    average similarity from SMILES from a sample

    :param smilies: list of SMILES
    :param ref_fps: reference fingerprints
    """

    mols = filter(lambda m: m, [Chem.MolFromSmiles(smiles) for smiles in smilies])
    fps = [Chem.RDKFingerprint(mol) for mol in mols]

    sims = []

    for ref_fp in ref_fps:
        sims.append(np.array(DataStructs.BulkTanimotoSimilarity(ref_fp, fps)))

    similarities = np.array(sims).mean(axis=0)

    return similarities
