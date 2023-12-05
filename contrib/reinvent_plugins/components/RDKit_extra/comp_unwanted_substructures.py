"""Use RDKit's filter catalogs to filter for unwanted structures"""

__all__ = ["UnwantedSubstructures"]

import sys
from dataclasses import dataclass
from typing import List
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from ..component_results import ComponentResults
from ..add_tag import add_tag
from reinvent_plugins.mol_cache import molcache

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    catalogs: List[List[str]]


@add_tag("__component", "filter")
class UnwantedSubstructures:
    def __init__(self, params: Parameters):
        filter_params = FilterCatalogParams()

        for catalogs in params.catalogs:
            for catalog_name in catalogs:
                try:
                    catalog = getattr(FilterCatalogParams.FilterCatalogs, catalog_name)
                except AttributeError:
                    msg = (
                        f"Unkbown catalog {catalog_name}: choose from "
                        f"{', '.join(FilterCatalogParams.FilterCatalogs.names.keys())}"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                filter_params.AddCatalog(catalog)

        self.catalog = FilterCatalog(filter_params)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for mol in mols:
            if not mol:
                score = np.nan
            else:
                score = 1 - self.catalog.HasMatch(mol)

            scores.append(score)

        return ComponentResults([np.array(scores, dtype=int)])
