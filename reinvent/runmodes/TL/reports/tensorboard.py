"""Write out a TensorBoard report"""

from __future__ import annotations
from typing import List, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from scipy.stats import entropy
from rdkit import Chem, DataStructs

from reinvent.runmodes.utils import make_grid_image

if TYPE_CHECKING:
    from reinvent.models import ModelAdapter

ROWS = 3
COLS = 10


@dataclass
class TBData:
    epoch: int
    mean_nll: float
    sample_batch_size: int
    ref_fps: List
    mean_nll_valid: float = None


def write_report(reporter, data, model: ModelAdapter, is_reinvent: bool, dataloader: None) -> None:
    """Write out TensorBoard data

    :param reporter: TB reporter for writing out the data
    :param data: data to be written out
    :param model: model adapter
    :param is_reinvent: whether this is Reinvent
    """

    # FIXME: check if this makes sense for Mol2Mol
    if is_reinvent:
        # NOTE: works only for Reinvent and Mol2Mol, implementation of
        # sample_smiles() specifically made for this purpose
        smilies, sample_nlls = model.sample_smiles(dataloader, num=data.sample_batch_size)

        similarities, kl_div = compute_similarity_from_sample(smilies, data.ref_fps)

        mean_nll_stats = {
            "Training Loss": data.mean_nll,
            "Sample Loss": sample_nlls.mean(),
        }

        if data.mean_nll_valid is not None:
            mean_nll_stats["Validation loss"] = data.mean_nll_valid

        reporter.add_scalars(
            "Mean loss (NLL)",
            mean_nll_stats,
            data.epoch,
        )

        reporter.add_scalar("KL divergence", kl_div, data.epoch)

        reporter.add_histogram("Tanimoto similarity on RDKitFingerprint", similarities, data.epoch)

        # FIXME: rows and cols depend on sample_batch_size
        image_tensor = make_grid_image(smilies, sample_nlls, "NLL", ROWS * COLS, ROWS)

        if image_tensor is not None:
            reporter.add_image(
                f"First {ROWS * COLS} Structures",
                image_tensor,
                data.epoch,
                dataformats="CHW",
            )  # channel, height, width
    else:
        mean_nll_stats = {
            "Training Loss": data.mean_nll,
        }

        if data.mean_nll_valid is not None:
            mean_nll_stats["Validation Loss"] = data.mean_nll_valid

        reporter.add_scalars(
            "Mean loss (NLL)",
            mean_nll_stats,
            data.epoch,
        )


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

    missing = len(smilies) - len(similarities)

    # FIXME: check if minimum mode is really appropriate
    similarities = np.pad(similarities, (0, missing), mode="minimum")

    if not hasattr(compute_similarity_from_sample, "first_similarities"):
        compute_similarity_from_sample.first_similarities = similarities

    kl_div = entropy(compute_similarity_from_sample.first_similarities, similarities)

    return similarities, kl_div
