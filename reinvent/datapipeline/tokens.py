"""Get SMILES tokens from a SMILES file or CSV"""

import os
import sys

import polars as pl

from reinvent.datapipeline.filters.regex import SMILES_TOKENS_REGEX


fn = sys.argv[1]
ext = os.path.splitext(fn)[1]

if ext == ".smi":
    df = pl.read_csv(fn, has_header=False, new_columns=["canonical_smiles"])
else:
    df = pl.read_csv(fn, separator="\t", columns=["canonical_smiles"])

print(f"Total number of SMILES: {len(df)}")

found_tokens = set()

for smiles in df["canonical_smiles"]:
    tokens = SMILES_TOKENS_REGEX.findall(smiles)
    found_tokens.update(tokens)

print(f"Found tokens: {sorted(found_tokens)}")
