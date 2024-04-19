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


def set_torch_device(args_device: str = None, device: str = None) -> torch.device:
    """Set the Torch device

    :param args_device: device name from the command line
    :param device: device name from the config
    """

    logger.debug(f"{device=} {args_device=}")

    # NOTE: ChemProp > 1.5 would need "spawn" but hits performance 4-5 times
    #       Windows requires "spawn"
    #torch.multiprocessing.set_start_method('fork')

    if args_device:  # command line overwrites config file
        # NOTE: this will throw a RuntimeError if the device is not available
        torch.set_default_device(args_device)
        actual_device = torch.device(args_device)
    elif device:
        torch.set_default_device(device)
        actual_device = torch.device(device)
    else:  # we assume there are no other devices...
        torch.set_default_device("cpu")
        actual_device = torch.device("cpu")

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
