"""Filter REINVENT CSV and extract top poses

Various filter options to filter down the initally generated molecules.  The
docked poses from AutoDockGPU are the extracted for this subset.
"""

import sys

import pandas as pd
from rdkit import Chem

SMILES_PROP = "smiles"


df = pd.read_csv(sys.argv[1])
print(f"original SMILES: {len(df)}")

df = df[df.SMILES_state > 0]
print(f"valid SMILES: {len(df)}")

df = df.drop_duplicates(subset=["SMILES"])
print(f"deduplicated SMILES: {len(df)}")

raw_columns = [col for col in df.columns if "(raw)" in col]
columns = [s.removesuffix(" (raw)") for s in raw_columns]
threshold = 0.7
cond = (df[columns] > threshold).all(axis=1)
df = df[cond]
print(f"SMILES above {threshold=}: {len(df)}")

max_dG = -14.0
top = df[df["AutoDockGPU (raw)"] < max_dG]
print(f"SMILES above {max_dG} kcal/mol: {len(top)}")

top.to_csv("top.csv", index=False)

mols = []

for smiles, step in top[["SMILES", "step"]].itertuples(index=False):
    file_step = step - 1

    if file_step > 0:
        num = "-" + str(file_step)
    else:
        num = ""

    suppl = Chem.SDMolSupplier(f"poses/best_pose{num}.sdf")

    for mol in suppl:
        if mol:
            props = mol.GetPropsAsDict()

            if SMILES_PROP in props and smiles == props[SMILES_PROP]:
                mols.append(mol)

with Chem.SDWriter(sys.argv[2]) as sdf:
  for mol in mols:
    sdf.write(mol)
