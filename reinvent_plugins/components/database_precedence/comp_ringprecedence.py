from typing import List, Literal

from pydantic.dataclasses import dataclass
from pydantic import Field


from rdkit.Chem import Mol
import numpy as np
import json

from .uru_ring_system_finder import get_rings, make_rings_generic
from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag

# __all__ = ["RingPrecedence"]


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    database_file: List[str]  # path to preproccessed ring precedence file
    nll_method: List[
        Literal["total", "max"]
    ]  # NLL computation method, either least probable ring or total
    make_generic: List[bool] = Field(default_factory=lambda: [False])


def compute_ring_nll(ring: str, nll_dictionary: dict[str:float], threshold=100.0) -> float:
    if ring in nll_dictionary:
        return nll_dictionary[ring]
    else:  # ring not precendented
        return threshold


@add_tag("__component")
class RingPrecedence:
    """
    Scoring component based on Pat Walter's ring extraction code
    Estimate the (negative log) likelihood of the ring systems in a molecule
    based on empirical probabilities from a database file.
    Database file should deseriaize to a dictionary containing two dictionaries:
        - "rings": maps specific ring SMILES to NLL values
        - "generic_rings": maps generic ring SMILES to NLL values

    Parameters:
        - database_file: Path to the database file containing NLL values.
        - nll_method: Scoring method, either "total" (sum of NLLs for all rings) or "max" (maximum NLL among rings).
        - make_generic: use generic rings (all atom/bond types removed); otherwise, uses the full smiles
    Returns:
        - scores: A numpy array of NLL scores for each molecule.
        - metadata: A dictionary containing "highest_nll_ring", the ring with the highest NLL for each molecule.


    Example database format:

    {
    rings: {ring_1_smiles:nll_1, ring_2_smiles:nll_2},
    generic_rings: {generic_ring_1_smiles:nll_1, generic_ring_2_smiles:nll_2}
     }
    E.g. from CHEMBL
    {"rings": {"c1ccccc1": 0.9884668635576631, "c1ccncc1": 2.9117327924945817,...},
    "generic_rings": {"C1CCCCC1": 0.7977199282489262, "C1CCCC1": 1.7240993251931773...}
    }

    Parameters:
        - database_file: Path to the database file containing NLL values.
        - nll_method: Scoring method, either "total" (sum of NLLs for all rings) or "max" (maximum NLL among rings).
        - make_generic: If True, uses generic ring representations for scoring; otherwise, uses specific rings.

    Returns:
        - scores: A numpy array of NLL scores for each molecule.
        - metadata: A dictionary containing "highest_nll_ring", the ring with the highest NLL for each molecule.

    Example database format:
    {
        "rings": {"c1ccccc1": 0.98, "c1ccncc1": 2.91, ...},
        "generic_rings": {"C1CCCCC1": 0.79, "C1CCCC1": 1.72, ...}
    }
    """

    def __init__(self, params: Parameters):
        self.database = json.load(open(params.database_file[0], "r"))
        self.make_generic = params.make_generic[0]
        self.nll_method = params.nll_method[0]

    def __call__(self, smilies: List[str]) -> ComponentResults:
        empirical_ring_nlls, metadata = self._compute_scores(smilies)
        return ComponentResults(empirical_ring_nlls, metadata=metadata)

    @molcache
    def _compute_scores(self, mols: List[Mol]) -> np.array:
        empirical_ring_nlls = []

        # extract rings
        all_rings = [get_rings(mol) for mol in mols]

        if self.make_generic:
            all_rings = [make_rings_generic(rings) for rings in all_rings]
            nll_dictionary = self.database["generic_rings"]
        else:
            nll_dictionary = self.database["rings"]
        ## compute empirical nll for each ring in each molecule and also log the highest nll ring for each molecule
        empirical_ring_nll_scores = []
        highest_nll_rings = []
        for rings in all_rings:
            empirical_ring_nlls = [compute_ring_nll(ring, nll_dictionary) for ring in rings]
            if empirical_ring_nlls:
                max_nll = np.max(empirical_ring_nlls)
                max_ring = rings[np.argmax(empirical_ring_nlls)]
            else:
                max_nll = None
                max_ring = None
            if self.nll_method == "total":
                empirical_ring_nll_scores.append(np.sum(empirical_ring_nlls))
            elif self.nll_method == "max":
                empirical_ring_nll_scores.append(max_nll)
            highest_nll_rings.append(max_ring)

        empirical_ring_nll_scores = np.array(empirical_ring_nll_scores)

        return [empirical_ring_nll_scores], {"highest_nll_ring": highest_nll_rings}
