"""Compute scores based on the similarity of generated molecules compared to a reference dataset"""

from __future__ import annotations
from rdkit import Chem

__all__ = ["ProtacValidity"]
from typing import List
import logging
import numpy as np
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    validation_datasets: List[str]


@add_tag("__component")
class ProtacValidity:
    """
    Calculates the number of target atoms in a SMILES and gives reward accordingly.
    Gives reward of 0 if given SMILES is invalid molecule.
    DISCLAIMER: Counts both aromatic and aliphatic atoms.
    """

    def __init__(self, params: Parameters):
        self.validation_datasets = params.validation_datasets
        self.smiles_type = "rdkit_smiles"

        self.metrics = {
            "Length"      : len,
            "MolWt"       : Descriptors.MolWt,
            "LogP"        : Descriptors.MolLogP,
            "TPSA"        : Descriptors.TPSA,
            "HBA"         : Descriptors.NumHAcceptors,
            "HBD"         : Descriptors.NumHDonors,
            "RotBonds"    : Descriptors.NumRotatableBonds,
            "NumCycles"   : Descriptors.RingCount,
            #"test"        : lambda mol: rdMolDescriptors.CalcNumAtomStereoCenters(mol) + rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        }

        self.weights = {name : 1/len(self.metrics) for name, _ in self.metrics.items()}

        dataframes = []
        for dataset_path in self.validation_datasets:
            try:
                dataframes.append(pd.read_csv(dataset_path, header=None, names=["SMILES"]))
            except Exception as e:
                print(e)
                continue

        df = pd.concat(dataframes, ignore_index=True)

        smiles_list = df["SMILES"]

        smiles_props = pd.DataFrame(columns=list(self.metrics.keys()))

        for i, smi in enumerate(smiles_list):
            row = []
            mol = Chem.MolFromSmiles(smi)
            row.append(len(smi))
            for name, desc in self.metrics.items():
                if name != "Length":
                    row.append(desc(mol))
            
            smiles_props.loc[i] = row
        
        self.stats_df = smiles_props.agg(['mean', 'std']).T

    @normalize_smiles
    def __call__(self, smilies):

        score_array = []

        for smi in smilies:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                score_array.append(np.nan)
                continue
        
            score = 0
            for name, metric_fn in self.metrics.items():
                if name != "Length":
                    value = metric_fn(smi)
                else:
                    value = metric_fn(mol)
                
                lb = self.stats_df.loc[name, 'mean'] - 1.96 * self.stats_df.loc[name, 'std']
                ub = self.stats_df.loc[name, 'mean'] + 1.96 * self.stats_df.loc[name, 'std']
                if value > lb and value < ub:
                    score += self.weights[name] * 1

            score_array.append(float(score))
    

        scores = [np.array(score_array, dtype=float)]

        return ComponentResults(scores=scores)