import os
import pandas as pd
from reinvent_plugins.components.comp_atomcount import AtomCount, Parameters
import numpy as np

# Define the working directory
wd = f"{os.getcwd()}/runs"

scorer = AtomCount(Parameters(target=["C", "O"]))

# Set working directory to the latest subdirectory in runs
subdirs = [d for d in os.listdir(wd) if os.path.isdir(os.path.join(wd, d))]
latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(wd, d)))
wd = os.path.join(wd, latest_subdir)

# Find latest stage subdirectory
subdirs = [d for d in os.listdir(wd) if os.path.isdir(os.path.join(wd, d))]
latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(wd, d)))

eval_file = os.path.join(wd, latest_subdir, "rl_1.csv")
print("Loading following csv:", eval_file)

df = pd.read_csv(eval_file)
df.insert(len(df.columns), 'AtomCountScore', np.nan)
for index, row in df.iterrows():
    smiles = row['SMILES']
    try:
        score = scorer([smiles]).scores[0][0]
    except Exception as e:
        score = np.nan
    df.at[index, 'AtomCountScore'] = score

df.to_csv("eval.csv", index=False)
