"""RDKit 2D Pharmacophore Fingerprints"""

from __future__ import annotations

__all__ = ["Pharm2DFP"]

import os
from dataclasses import dataclass
from typing import List
import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
import numpy as np

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag

logger = logging.getLogger("reinvent")

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "features")


@add_tag("__parameters")
@dataclass
class Parameters:
    ref_smiles: List[str]
    feature_definition: List[str]  # base, minimal, gobbi
    bins: List[List[int]]
    min_point_count: List[int]
    max_point_count: List[int]
    similarity: List[str]
    similarity_params: List[dict]


@add_tag("__component")
class Pharm2DFP:
    def __init__(self, params: Parameters):
        self.ref_fps = []
        self.signature_factories = []
        self.similarities = []
        self.similarities_params = []

        for smiles, fdef, bins, minp, maxp, sim, sim_params in zip(
            params.ref_smiles,
            params.feature_definition,
            params.bins,
            params.min_point_count,
            params.max_point_count,
            params.similarity,
            params.similarity_params,
        ):
            fdef_name = fdef.capitalize()

            if fdef_name == "Gobbi":
                signature_factory = Gobbi_Pharm2D.factory
            else:
                fdef_filename = os.path.join(FEATURE_DIR, f"{fdef_name}Features.fdef")

                feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_filename)
                signature_factory = SigFactory(
                    feature_factory, minPointCount=minp, maxPointCount=maxp
                )

                b = iter(bins)
                signature_factory.SetBins(list(zip(b, b)))
                signature_factory.Init()

            self.signature_factories.append(signature_factory)

            mol = Chem.MolFromSmiles(smiles)

            if not mol:
                raise RuntimeError(f"{__name__}: invalid SMILES {smiles}")

            fp = Generate.Gen2DFingerprint(mol, signature_factory)  # replace
            self.ref_fps.append(fp)

            sim_name = sim.capitalize()

            try:
                self.similarities.append(getattr(DataStructs, f"Bulk{sim_name}Similarity"))
                self.similarities_params.append(sim_params)
            except:
                raise RuntimeError(f"{__name__}: {sim_name} similarity not supported by RDKit")

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []

        for ref_fp, signature_factory, similarity, sim_params in zip(
            self.ref_fps, self.signature_factories, self.similarities, self.similarities_params
        ):
            target_fps = []

            for mol in mols:
                target_fps.append(Generate.Gen2DFingerprint(mol, signature_factory))

            scores.append(np.array(similarity(ref_fp, target_fps, **sim_params), dtype=float))

        return ComponentResults(scores)
