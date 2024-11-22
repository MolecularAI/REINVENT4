"""Prior registry

Maps a key to an actual prior filename.
"""

__all__ = ["prior_registry"]
import os
import pathlib

import reinvent


if "REINVENT_PRIOR_BASE" in os.environ:
    PRIOR_BASE = pathlib.Path(os.environ["REINVENT_PRIOR_BASE"])
else:
    PRIOR_BASE = pathlib.Path(reinvent.__file__).parents[1] / "priors"

prior_registry = {
    ".reinvent": PRIOR_BASE / "reinvent.prior",
    ".libinvent": PRIOR_BASE / "libinvent.prior",
    ".linkinvent": PRIOR_BASE / "linkinvent.prior",
    ".m2m_high": PRIOR_BASE / "mol2mol_high_similarity.prior",
    ".m2m_medium": PRIOR_BASE / "mol2mol_medium_similarity.prior",
    ".m2m_mmp": PRIOR_BASE / "mol2mol_mmp.prior",
    ".m2m_scaffold": PRIOR_BASE / "mol2mol_scaffold.prior",
    ".m2m_scaffold_generic": PRIOR_BASE / "mol2mol_scaffold_generic.prior",
    ".m2m_similarity": PRIOR_BASE / "pubchem_ecfp4_with_count_with_rank_reinvent4_dict_voc.prior",
}
