"""Matched molecular pairs"""

from __future__ import annotations

__all__ = ["MMP"]

import logging
import shlex
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag
from ..normalize import normalize_smiles

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

    reference_smiles: List[List[str]]
    num_of_cuts: List[int] = Field(default_factory=lambda: [1])
    max_variable_heavies: List[int] = Field(default_factory=lambda: [40])
    max_variable_ratio: List[float] = Field(default_factory=lambda: [0.33])


FRAG_CMD = "mmpdb --quiet fragment --num-cuts {ncuts}"
IDX_CMD = (
    "mmpdb --quiet index --out csv --symmetric --max-variable-heavies {heavy} "
    "--max-variable-ratio {ratio}"
)


@add_tag("__component")
class MMP:
    def __init__(self, params: Parameters):
        self.ref_smilies = params.reference_smiles
        self.num_of_cuts = params.num_of_cuts
        self.max_variable_heavies = params.max_variable_heavies
        self.max_variable_ratio = params.max_variable_ratio

        # needed in the normalize_smiles decorator
        # FIXME: really needs to be configurable for each model separately
        self.smiles_type = "rdkit_smiles"

        self.number_of_endpoints = len(params.reference_smiles)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        scores = []

        self.ref_smilies = [
            [
                Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
                for smi in self.ref_smilies[0]
                if Chem.MolFromSmiles(smi)
            ]
        ]
        smilies = [
            Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
            for smi in smilies
            if Chem.MolFromSmiles(smi)
        ]

        for ref_smilies, ncuts, max_heavy, max_ratio in zip(
            self.ref_smilies, self.num_of_cuts, self.max_variable_heavies, self.max_variable_ratio
        ):
            smiles_csv = format_smilies(smilies, ref_smilies)

            frag_cmd = FRAG_CMD.format(ncuts=ncuts)
            result1 = run_command(shlex.split(frag_cmd), input=smiles_csv)
            if result1.returncode != 0:
                logger.warning(
                    f"MMP process returned non-zero returncode ({result1.returncode})."
                    f" Stderr:\n{result1.stderr}"
                )

            idx_cmd = IDX_CMD.format(heavy=max_heavy, ratio=max_ratio)
            result2 = run_command(shlex.split(idx_cmd), input=result1.stdout)
            if result2.returncode != 0:
                logger.warning(
                    f"MMP process returned non-zero returncode ({result2.returncode})."
                    f" Stderr:\n{result2.stderr}"
                )

            data = parse_index_output(result2.stdout)
            scores.append(get_scores(data, smilies, ref_smilies))

        return ComponentResults(scores)


def format_smilies(in_smilies: List[str], ref_smilies: List[str]) -> str:
    """Format SMILES and reference SMILES for MMPDB

    :param in_smilies: input SMILES
    :param ref_smilies: reference SMILES
    :returns: CSV formatted string
    """

    data = ["SMILES ID"]

    for i, smiles in enumerate(ref_smilies):
        data.append(f"{smiles} Source_ID_{i + 1}")

    for i, smiles in enumerate(in_smilies):
        data.append(f"{smiles} Generated_ID_{i + 1}")

    return "\n".join(data)


def parse_index_output(index_output: str) -> pd.DataFrame | None:
    """Parse output from MMPDB index

    :param index_output: output from MMPDB index command
    :returns:
    """

    if not index_output:
        return None

    data = pd.read_csv(
        StringIO(index_output),
        sep="\t",
        header=None,
        names=[
            "Source_Smi",
            "Target_Smi",
            "Source_Mol_ID",
            "Target_Mol_ID",
            "Transformation",
            "Core",
        ],
    )

    data = data[
        (data["Source_Mol_ID"].str.contains("Source_ID"))
        & (data["Target_Mol_ID"].str.contains("Generated_ID"))
    ]

    data["Source_R_len"] = data["Transformation"].apply(len)
    data = data.sort_values("Source_R_len")
    data.drop_duplicates(subset=["Source_Mol_ID", "Target_Mol_ID"], inplace=True)

    return data


def get_scores(df: pd.DataFrame, in_smilies: List[str], ref_smilies: List[str]) -> np.array:
    """Extract scores from MMP parsed dataframe

    :param df: pandas dataframe with MMP parsed data
    :param in_smilies: input SMILES
    :param ref_smilies: reference SMILES
    :returns: scores
    """

    if df is None or df.empty:
        return np.full(len(in_smilies), "No MMP")

    mmp_result = []

    for smiles in in_smilies:
        mmp_ref_list = []

        for ref_smiles in ref_smilies:
            if len(df[(df["Source_Smi"] == ref_smiles) & (df["Target_Smi"] == smiles)]) > 0:
                mmp_ref_list.append(True)
            else:
                mmp_ref_list.append(False)

        mmp_result.append("MMP" if any(mmp_ref_list) else "No MMP")

    return np.array(mmp_result)
