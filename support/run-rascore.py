#!/bin/env python3
#
# This is an example how to use the ExternalProcess scoring component using
# RASCore.  The scripts expects a list of SMILES from stdin and will
# write a JSON string to stdout.
#
# Retrosynthetic Accessibility Score (https://doi.org/10.1039/D0SC05401A)
# with code from https://github.com/reymond-group/RAscore.
#
# Setup shown for scoring file include feature: scoring.filename
#
# [[component.ExternalProcess.endpoint]]
# name = "RAScore"
# weight = 0.6
#
# # The software requires a specific virtual environment based on Python 3.7/8
# # The --no-capture-output is necessary to pass through stdout from REINVENT4
# params.executable = "/home/user/miniconda3/condabin/mamba"
# params.args = "run --no-capture-output -n rascore /home/user/projects/RAScore/run-rascore.py
# # No transform needed as score is already between 0 and 1
#


import sys
import json

from RAscore import RAscore_NN


# Created from default model in the repository:
# from tensorflow import keras
# model = keras.models.load_model("models/DNN_chembl_fcfp_counts/model.tf")
# model.save("/home/user/projects/RAScore/new_tf_2.5")
RASCORE_MODEL = "/home/user/projects/RAScore/new_tf_2.5"

smilies = [smiles.strip() for smiles in sys.stdin]

# Everything from here to END is specific to the scorer

# The default model will give a warning regarding the tensorflow version
# nn_scorer = RAscore_NN.RAScorerNN()
nn_scorer = RAscore_NN.RAScorerNN(RASCORE_MODEL)
scores = []

for smiles in smilies:
    score = nn_scorer.predict(smiles)  # returns numpy.float32
    scores.append(float(score))

# END


# Format the JSON string for REINVENT4 and write it to stdout
data = {"version": 1, "payload": {"predictions": scores}}

print(json.dumps(data))
