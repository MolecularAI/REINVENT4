import json
import os
import tempfile
from typing import Tuple

import pandas as pd
import yaml

from reinvent_plugins.components.synthsense.parameters import ComponentLevelParameters


__ENV_NAME = "CAZP_PROFILES"
CAZP_PROFILE = __ENV_NAME if __ENV_NAME in os.environ else "HITINVENT_PROFILES"


def mergedicts(a: dict, b: dict, path=None) -> dict:
    """Merges b into a, in place"""
    # https://stackoverflow.com/a/7205107
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                mergedicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                formatted_path = ".".join(path + [str(key)])
                raise Exception(
                    f"Conflict at {formatted_path}: {a[key]} vs {b[key]}." f" Full dicts: {a}, {b}."
                )
        else:
            a[key] = b[key]
    return a


def prepare_config(params: ComponentLevelParameters) -> Tuple[dict, str]:
    """Prepare AiZynthFinder config and command.

    This function constructs AiZynthFinder config, in the following order:
    - Lowest priority is base config in the profiles.
    - Next priority are values in profiles, they overwrite base config.
    - Highest priority are inline values in Reinvent scoring component.

    :param params: Reinvent scoring component params
    :return: dict with AiZynthFinder-ready config
    """

    if CAZP_PROFILE in os.environ:
        uiprofiles_file = os.environ[CAZP_PROFILE]
        with open(uiprofiles_file) as fp:
            uiprofiles = json.load(fp)
    else:
        raise ValueError(
            f"Missing CAZP Profiles file specified by environmental variable {CAZP_PROFILE}."
        )

    if "base_aizynthfinder_config" in uiprofiles:
        base_aizynthfinder_config = uiprofiles.get("base_aizynthfinder_config")
    else:
        raise ValueError(
            f"Missing base_aizynthfinder_config in CAZP profiles file"
            f" specified by environmental variable {CAZP_PROFILE}."
        )

    if "custom_aizynth_command" in uiprofiles:
        cmd = uiprofiles.get("custom_aizynth_command")
    else:
        raise ValueError(
            f"Missing custom_aizynth_command in CAZP profiles file"
            f" specified by environmental variable {CAZP_PROFILE}."
        )

    with open(base_aizynthfinder_config) as fp:
        config = yaml.safe_load(fp)

    if params.stock_profile:
        if params.stock_profile not in uiprofiles["stock_profiles"]:
            raise ValueError(
                f"Stock profile {params.stock_profile} not found,"
                f" available: {uiprofiles.get('stock_profiles', {}).keys()}"
            )
        mergedicts(
            config,
            uiprofiles["stock_profiles"][params.stock_profile]["config"],
        )

    if params.reactions_profile:
        if params.reactions_profile not in uiprofiles["reactions_profiles"]:
            raise ValueError(
                f"Reactions profile {params.reactions_profile} not found,"
                f" available: {uiprofiles.get('reactions_profiles', {}).keys()}"
            )
        mergedicts(
            config,
            uiprofiles["reactions_profiles"][params.reactions_profile]["config"],
        )

    # AiZynthFinder v4 uses "search" instead of "properties".
    if "search" in config:
        properties_key = "search"
    elif "properties" in config:
        properties_key = "properties"
    else:
        raise ValueError("Neither properties (v3) nor search (v4) in AiZynth Config.")

    if params.number_of_steps is not None:
        config[properties_key]["max_transforms"] = params.number_of_steps

    if params.time_limit_seconds is not None:
        config[properties_key]["time_limit"] = params.time_limit_seconds

    # Add custom stock file (.smi or .csv).
    # v3: {"stock": {"files": {"CUSTOM": "stock1.smi"}}}
    # v4: {"stock": {"CUSTOM": "stock1.smi"}}
    if params.stock:
        mergedicts(config, {"stock": params.stock})

    # Add scorer for custom stock.
    # "scorer": {"StockAvailabilityScorer": {"source_score": {"CUSTOM": 1 } } }
    if params.scorer:
        mergedicts(config, {"scorer": params.scorer})

    return config, cmd


def smi2inchikey(smi: str) -> str:
    """Convert SMILES string to InChI key."""

    from rdkit.Chem import MolFromSmiles, MolToInchiKey

    mol = MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot convert smiles to mol: {smi}")
    return MolToInchiKey(mol)


def convert_smi_file_to_csv_file(smi_file, tmpdir):
    with tempfile.NamedTemporaryFile(
        "wt",
        delete=False,
        prefix="aizynth-custom-stock-inchikey-",
        suffix=".csv",
        dir=tmpdir,
    ) as inchikeycsv:
        df = pd.read_csv(smi_file, sep=r"\s+", header=None, names=["smiles"])
        df["inchi_key"] = df["smiles"].apply(smi2inchikey)
        df.to_csv(inchikeycsv.name, index=False)
    return inchikeycsv.name


def ensure_custom_stock_is_inchikey(config, tmpdir):
    """Convert .smi stock with smiles to .csv stock with inchi_key."""

    # This is for v3
    # Example YAML:
    # stock:
    #   files:
    #     stock1: stock1.smi
    for stock_name, stock_file in config.get("stock", {}).get("files", {}).copy().items():
        if stock_file.endswith(".smi"):
            stock_file_csv = convert_smi_file_to_csv_file(stock_file, tmpdir)
            config["stock"]["files"][stock_name] = stock_file_csv

    # This is for v4
    # Example YAML:
    # stock:
    #   stock1: stock1.smi
    #   stock2:
    #     path: stock2.smi
    for stock_name, stock_data in config["stock"].copy().items():
        if isinstance(stock_data, str) and stock_data.endswith(".smi"):
            stock_file_csv = convert_smi_file_to_csv_file(stock_data, tmpdir)
            config["stock"][stock_name] = stock_file_csv
        elif isinstance(stock_data, dict) and stock_data.get("path", "").endswith(".smi"):
            stock_file_csv = convert_smi_file_to_csv_file(stock_data["path"], tmpdir)
            config["stock"][stock_name]["path"] = stock_file_csv
