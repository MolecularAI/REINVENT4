from __future__ import annotations

__all__ = ["canonicalize_smiles"]
from typing import TYPE_CHECKING
import logging

from reinvent.chemistry import conversions

if TYPE_CHECKING:
    from reinvent_scoring.scoring.score_summary import FinalSummary


logger = logging.getLogger(__name__)


def canonicalize_smiles(score_summary: FinalSummary, rdkit_smiles_flags: dict) -> None:
    """Canonicalize all valid SMILES

    NOTE: this will modify the original data structure

    :param score_summary: score summary object
    :param rdkit_smiles_flags: canonicalization flags for RDKit
    """

    smilies = score_summary.scored_smiles

    for i in score_summary.valid_idxs:
        # FIXME: need to control what happens here in RDKit, by default
        #        sanitize=true will also kekulize the SMILES which may lead
        #        to an invalid SMILES when read again
        smilies[i] = conversions.convert_to_rdkit_smiles(smilies[i], **rdkit_smiles_flags)

    logger.debug(f"total SMILES={len(smilies)}; valid SMILES={len(score_summary.valid_idxs)}")
