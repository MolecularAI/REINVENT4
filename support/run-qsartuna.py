#!/bin/env python3
#
# This is an example how to use the ExternalProcess scoring component using
# QSARtuna, see DOI 10.1021/acs.jcim.4c00457.  The scripts expects a list of
# SMILES from stdin and will # write a JSON string to stdout.
#
# QSARtuna code at https://github.com/MolecularAI/QSARtuna.
#
# [[component.ExternalProcess.endpoint]]
# name = "QSARtuna model"
# weight = 0.6
#
# # Run Qptuna in its own environment
# # The --no-capture-output is necessary to pass through stdout from REINVENT4
# params.executable = "/home/user/miniconda3/condabin/mamba"
# params.args = "run --no-capture-output -n qsartuna /path/to/run-qsartuna.py model_filename
#
# # Don't forget the transform if needed!
#


import sys
import json
import pickle


smilies = [smiles.strip() for smiles in sys.stdin]

# Everything from here to END is specific to the scorer

with open(sys.argv[1], "rb") as mfile:
    model = pickle.load(mfile)

scores = model.predict_from_smiles(smilies, uncert=False)

# END


# Format the JSON string for REINVENT4 and write it to stdout
data = {"version": 1, "payload": {"predictions": list(scores)}}

print(json.dumps(data))
