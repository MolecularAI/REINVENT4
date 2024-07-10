#!/bin/env python3
#
# This is an example how to use the ExternalProcess scoring component using
# ChemProp.  The scripts expects a list of SMILES from stdin and will
# write a JSON string to stdout.
#
# Setup shown for scoring file include feature: scoring.filename
#
# [[component.ExternalProcess.endpoint]]
# name = "ChemProp"
# weight = 0.6
#
# # If already in the right conda environment the script could be run directly
# # The --no-capture-output is necessary to pass through stdout from REINVENT4
# params.executable = "/home/user/miniconda3/condabin/conda"
# params.args = "run --no-capture-output -n dev /home/user/projects/reinvent/run-chemprop.py"
#
# transform.type = "reverse_sigmoid"
# transform.high = -0.0
# transform.low = -20.0
# transform.k = 0.4
#


from os import devnull
import sys
import json
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import chemprop
import numpy as np


# Read the SMILES from stdin
smilies = [[smiles.strip()] for smiles in sys.stdin]


# Everything from here to END is specific to the scorer


# Annoyingly ChemProp babbles too much on stdout so suppress this output
@contextmanager
def suppress_output():
    """Context manager to redirect stdout and stderr to /dev/null"""

    with open(devnull, "w") as nowhere:
        with redirect_stderr(nowhere) as err, redirect_stdout(nowhere) as out:
            yield (err, out)


# ChemProp parameters
checkpoint_dir = "ChemProp/models"
rdkit_2d_normalized = True

args = [
    "--checkpoint_dir",  # ChemProp models directory
    checkpoint_dir,
    "--test_path",  # required
    "/dev/null",
    "--preds_path",  # required
    "/dev/null",
]

if rdkit_2d_normalized:
    args.extend(["--features_generator", "rdkit_2d_normalized", "--no_features_scaling"])

# Run the SMILES through ChemProp
with suppress_output():
    chemprop_args = chemprop.args.PredictArgs().parse_args(args)
    chemprop_model = chemprop.train.load_model(args=chemprop_args)

    preds = chemprop.train.make_predictions(
        model_objects=chemprop_model,
        smiles=smilies,
        args=chemprop_args,
        return_invalid_smiles=True,
        return_uncertainty=False,
    )

# Collect the score for each SMILES
scores = [val[0] if "Invalid SMILES" not in val else np.nan for val in preds]

# END


# Format the JSON string for REINVENT4 and write it to stdout
data = {"version": 1, "payload": {"predictions": scores}}

print(json.dumps(data))
