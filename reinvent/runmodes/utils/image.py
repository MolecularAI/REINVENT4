"""Create a grid image from molecules for Tensorboard."""

from __future__ import annotations

__all__ = ["make_grid_image"]
from typing import Tuple
import logging

import torch
from torchvision import transforms
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

logger = logging.getLogger(__name__)
convert_img_to_tensor = transforms.ToTensor()


def make_grid_image(smilies, data, label: str, sample_size: int, nrows: int) -> Tuple(torch.Tensor, int) | None:
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
        subImgSize=(250, 250),
        legends=legends,
    )

    return convert_img_to_tensor(png_image), len(mols)
