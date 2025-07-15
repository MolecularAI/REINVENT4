# this script reads in a csv file containing smiles (in "smiles_column") and enumerates ring systems and their precdence,
# resultsing in a .json database that can be used in the "comp_ringprecedence" scoring component
# example use
# python compute_database_precendence.py --dataset [CSV_FILE] --smiles_column SMILES --output_file ring_databse.json --num_cpus 4

import argparse
import os
import json
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from reinvent_plugins.components.database_precedence.uru_ring_system_finder import (
    get_rings,
    make_rings_generic,
)


def clean_and_flatten_smiles(smiles):
    # Convert SMILES to RDKit Mol object
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Return None for invalid SMILES
    except:
        print(f" invalid smiles {smiles}")
        return None

    # Remove stereochemistry
    Chem.RemoveStereochemistry(mol)

    # Canonicalize SMILES
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles


def ring_from_smiles(smiles):
    # since the scoring component method works on mol objects, need to convert here
    return get_rings(Chem.MolFromSmiles(smiles))


# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="input dataset, csv file")
    parser.add_argument("--smiles_column", type=str, required=True, help="smiles column")
    parser.add_argument("--output_file", type=str, required=True, help="output file")
    parser.add_argument("--num_cpus", type=int, default=4, help="# of cpus to use in parallel")
    args = parser.parse_args()

    dataset = args.dataset
    smiles_column = args.smiles_column
    output_file = args.output_file
    num_cpus = args.num_cpus

    # read in data
    df = pd.read_csv(dataset)
    print(f" read in DF size {df.shape}")

    # Use a multiprocessing pool to cleaning SMILES and remove stereo
    with mp.Pool(num_cpus) as pool:
        df["cleaned_smiles"] = pool.map(clean_and_flatten_smiles, df[smiles_column])

    # Filter out invalid SMILES and rings
    df = df.loc[~df["cleaned_smiles"].isnull()]
    print(f"after cleaning, df size is  {df.shape}")

    # Use a multiprocessing pool to parallelize ring system detection
    with mp.Pool(num_cpus) as pool:
        df["rings"] = list(tqdm(pool.imap(ring_from_smiles, df["cleaned_smiles"]), total=len(df)))
    with mp.Pool(num_cpus) as pool:
        df["generic_rings"] = list(tqdm(pool.imap(make_rings_generic, df["rings"]), total=len(df)))

    df = df.loc[~df["rings"].isnull()]
    print(f"after rings computation, valid df size is {df.shape}")

    # flatten and analyze rings lists
    all_rings = [item for sublist in df["rings"] if isinstance(sublist, list) for item in sublist]
    all_rings_df = pd.DataFrame(pd.Series(all_rings)).value_counts().reset_index()
    all_rings_df.columns = ["ring", "count"]
    all_rings_df["probability"] = all_rings_df["count"] / all_rings_df["count"].sum()
    all_rings_df["nll"] = -1 * np.log(all_rings_df["probability"])

    # repeat for generic rings
    all_generic_rings = [
        item for sublist in df["generic_rings"] if isinstance(sublist, list) for item in sublist
    ]
    all_generic_rings_df = pd.DataFrame(pd.Series(all_generic_rings)).value_counts().reset_index()
    all_generic_rings_df.columns = ["ring", "count"]
    all_generic_rings_df["probability"] = (
        all_generic_rings_df["count"] / all_generic_rings_df["count"].sum()
    )
    all_generic_rings_df["probability"] = (
        all_generic_rings_df["count"] / all_generic_rings_df["count"].sum()
    )
    all_generic_rings_df["nll"] = -1 * np.log(all_generic_rings_df["probability"])
    print(all_generic_rings_df)
    nll_dict = {
        "rings": all_rings_df.set_index("ring")["nll"].to_dict(),
        "generic_rings": all_generic_rings_df.set_index("ring")["nll"].to_dict(),
    }

    # serialize output
    json.dump(nll_dict, open(output_file, "w"))
