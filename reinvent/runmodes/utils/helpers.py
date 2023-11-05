"""Some auxiliary functionality that does not fit anywhere else.

FIXME: may need to move some/all of these
"""

from __future__ import annotations

__all__ = ["disable_gradients", "set_torch_device", "join_fragments"]
import logging
from typing import List, TYPE_CHECKING

import torch
from rdkit import Chem

from reinvent.chemistry.library_design import BondMaker, AttachmentPoints

if TYPE_CHECKING:
    from rdkit import Chem
    from reinvent.models import ModelAdapter
    from reinvent.models.model_factory.sample_batch import SampleBatch


# FIXME: really just containers of static functions
attachment_points = AttachmentPoints()
bond_maker = BondMaker()
logger = logging.getLogger(__name__)


def disable_gradients(model: ModelAdapter) -> None:
    """Disable gradient tracking for all parameters in a model

    :param model: the model for which all gradient tracking will be switched off
    """

    for param in model.get_network_parameters():
        param.requires_grad = False


def set_torch_device(device: str = None, use_cuda: bool = True) -> torch.device:
    """Set the Torch device

    :param device: device name from the command line
    :param use_cuda: whether use_cuda was set in the user config
    """

    logger.debug(f"{device=} {use_cuda=}")

    if device:  # command line overwrites config file
        # NOTE: this will throw a RuntimeError if the device is not available
        actual_device = torch.device(device)
    elif use_cuda and torch.cuda.is_available():
        actual_device = torch.device("cuda")
    else:  # we assume there are no other devices...
        actual_device = torch.device("cpu")

    # FIXME: check if this can be replaced
    # See https://github.com/pytorch/pytorch/issues/82296
    if actual_device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:  # assume CPU...
        torch.set_default_tensor_type(torch.FloatTensor)

    logger.debug(f"{actual_device=}")

    return actual_device


def join_fragments(sequences: SampleBatch, reverse: bool, keep_labels: bool = False) -> List[Chem.Mol]:
    """Join two fragments: for LibInvent and LinkInvent

    :param sequences: a batch of sequences
    :param reverse: order of fragments  FIXME: needs better name!
    :returns: a list of RDKit molecules
    """

    mols = []

    for sample in sequences:
        if not reverse:  # LibInvent
            frag1 = sample.input
            frag2 = sample.output
        else:  # LinkInvent
            frag1 = sample.output
            frag2 = sample.input

        scaffold = attachment_points.add_attachment_point_numbers(frag1, canonicalize=False)
        mol: Chem.Mol = bond_maker.join_scaffolds_and_decorations(  # may return None
            scaffold, frag2, keep_labels_on_atoms=keep_labels
        )
        mols.append(mol)

    return mols
