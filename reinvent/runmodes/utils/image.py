"""Create a grid image from molecules for Tensorboard."""

from __future__ import annotations

__all__ = ["make_grid_image", "get_matching_substructure"]
from typing import TYPE_CHECKING
import logging

import torch
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

from reinvent.chemistry.logging import (
    padding_with_invalid_smiles,
    check_for_invalid_mols_and_create_legend,
    find_matching_pattern_in_smiles,
)

if TYPE_CHECKING:
    from reinvent_scoring.scoring.score_summary import FinalSummary

logger = logging.getLogger(__name__)
convert_img_to_tensor = transforms.ToTensor()


def make_grid_image(smilies, data, label: str, sample_size: int, nrows: int) -> torch.Tensor:
    """Create image grid from the SMILES

    :param smilies: score summary
    :param data: array with numbers the same length as smilies
    :param label: string label data
    :param sample_size: sample size
    :param nrows: number of rows
    :returns: an image in tensor format
    """

    mols = []
    legends = []

    for smiles, datum in zip(smilies, data):
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol:
            mols.append(mol)
            legends.append(f"{label}={datum:.2f}")

    if not mols:
        logger.debug("No valid RDKit molecules found for MolsToGridImage(): ignoring")
        return None

    # RDKit creates a PIL image
    png_image = MolsToGridImage(
        mols[:sample_size],
        molsPerRow=nrows,
        subImgSize=(350, 350),
        legends=legends,
    )

    return convert_img_to_tensor(png_image)


def get_matching_substructure(score_summary: FinalSummary):
    smarts_pattern = ""

    for summary_component in score_summary.scaffold_log:
        if summary_component.parameters.component_type == "matching_substructure":
            smarts = summary_component.parameters.specific_parameters.get("smilies", [])

            if len(smarts) > 0:
                smarts_pattern = smarts[0]

    return smarts_pattern
