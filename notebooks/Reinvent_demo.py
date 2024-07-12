# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # _De novo_ Reinforcement Learning with the Reinvent prior
#
# This is a short demo to
# - Set up a reinforcment learning run
# - Carry out a reinforcment learning run
# - Visualize the results with TensorBoard
# - Extract the raw data from TensorBoard and how to work with it

# +
import os
import shutil

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import reinvent
from reinvent.notebooks import load_tb_data, plot_scalars, get_image, create_mol_grid

# %load_ext tensorboard
# -

# ## Set up the REINVENT run
#
# A work directory is defined and created anew (all previous data deleted if the directory already exsists).  The prior file is taken from the REINVENT repository and the agent is initially the same as the prior.  A TOML configuration is written out to file in the work directory.

wd = "/tmp/R4_notebooks_output"

# ### Delete existing working directory (!)
#
# Skip this step if you have already carried out the RL run but wish to analyis the results further.

shutil.rmtree(wd, ignore_errors=True)
os.mkdir(wd)
os.chdir(wd)

# ### Global configuration parameters
#
# Parameters global to the run:
# - The run type: one of "scoring", "sampling", "transfer_learning", and "staged_learning"
# - The device to run on: "cpu" or "cuda:0" where the number is the device index (needed for ROCm)
# - The output directory for TensorBoard (optional)
# - The configuration file in JSON format (optional)

global_parameters = """
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "tb_stage1"
json_out_config = "_stage1.json"
"""

# ### Parameters
#
# Here we specify the model files, the prefix for the output CSV summary file and the batch size for sampling and stochastic gradient descent (SGD).  The batch size is often given in 2^N but there is in now way required.  Typically batch sizes are betwen 50 and 150.  Batch size effects on SGD and so also the learning rate.  Some experimentation may be required to adjust this but keep in mind that, say, raising the total score as fast as possible is not necessarily the best choice as this may hamper exploration.

# +
prior_filename = os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior")
agent_filename = prior_filename

parameters = f"""
[parameters]

prior_file = "{prior_filename}"
agent_file = "{agent_filename}"
summary_csv_prefix = "stage1"

batch_size = 100

use_checkpoint = false
"""
# -

# ### Reinforcement Learning strategy

learning_strategy = """
[learning_strategy]

type = "dap"
sigma = 128
rate = 0.0001
"""

# ###  Stage setup
#
# Here we only use a single stage. The aim of this stage is to create an agent which is highly likely to generate "drug-like" molecules (as per QED and Custom Alerts) with no stereocentres
#
# The stage will terminate when a maximum number of 300 steps is reached.  Termination could occur earlier when the maximum score of 1.0 is exceeded but this is very unlikely to occur.  A checkpoint file is written out which can be used as the agent in a subsequent stage.
#
# The scoring function is a weighted product of all the scoring components: QED and number of sterecentres.  The latter is used here to avoid stereocentres as they are not support by the Reinvent prior.  Zero stereocentres aids in downstream 3D task to avoid having to carry out stereocentre enumeration.  Custom alerts is a filter which filters out (scores as zero) all generated compounds which match one of the SMARTS patterns.  Number of sterecentres uses a transformation function to ensure the component score is between 0 and 1.

stages = """
[[stage]]

max_score = 1.0
max_steps = 300

chkpt_file = 'stage1.chkpt'

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"

params.smarts = [
    "[*;r8]",
    "[*;r9]",
    "[*;r10]",
    "[*;r11]",
    "[*;r12]",
    "[*;r13]",
    "[*;r14]",
    "[*;r15]",
    "[*;r16]",
    "[*;r17]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
]

[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.6


[[stage.scoring.component]]
[stage.scoring.component.NumAtomStereoCenters]

[[stage.scoring.component.NumAtomStereoCenters.endpoint]]
name = "Stereo"
weight = 0.4

transform.type = "left_step"
transform.low = 0
"""

# +
config = global_parameters + parameters + learning_strategy + stages

toml_config_filename = "stage1.toml"

with open(toml_config_filename, "w") as tf:
    tf.write(config)
# -

# ##  Start Reinforcement Learning
#
# This run will take several minutes to finish and timings are provided at the end of the run.  After the run the TensorBoard output can be used to inspect the results of the run

# %%time
# !reinvent -l stage1.log $toml_config_filename

# ### Inspect results with TensorBoard
#
# TensorBoard needs to be started after REINVENT has finished.  Scatter plots of all scoring components are shown (raw and transformed) in the _SCALARS_ tab as well as loss and fractions of valid and duplicate (per batch) SMILES.  The _IMAGES_ tab shows the first sampled molecules for each recorded RL step labelled with the total score for the molecule.
#
# The loss likehoods (negative log likelihoods, NLL) for the agent is expected to move away from the prior NLL and also have smaller NLLs than the prior.  This shows that the agent is increasingly producing molecules different from the prior, closer to the signal from the scoring function as requested.  For sample efficiency it is desirable to observed few duplicates and a high number of valid molecules.  Note that results are stochastic and will not be different in every run of RL, even with exactly the same configuration.

# %tensorboard --bind_all --logdir $wd/tb_stage1_0

# ## Extract data from TensorBoard
#
# TensorBoard data can be directy extracted as shown in this section.

# ### Load the TB data

ea = load_tb_data(wd)

# ### Plot all scalars
#
# All scalar values (except the raw components) are plotted here.  The data is also return as a Pandas dataframe and can so be conveniently used for further analysis or storing to a file.

df = plot_scalars(ea)

df

# ### Display an image from TB
#
# Shows (only) the last image from TB.  The image depicts the first 30 molecules generated in the very last step of RL:

img = get_image(ea)
display(img)

# ## Extract data from the CSV file
#
# The CSV file is generated during the RL run in real time i.e. as soon as the data is available it is written to file. So even if the RL job crashes in the middle of the run some partial data will be available.  The TensorBoard data is a subset of the data in the CSV file.  The CSV file contains in addition all SMILES strings, their state (0=invalid, 1=valid, 2=batch duplicate), the scaffold if a diversity filter has been used.

csv_file = os.path.join(wd, "stage1_1.csv")
df = pd.read_csv(csv_file)
df

# ### Sample efficiency
#
# Count the number of total invalid and duplcate SMILES and compare to the total number of generated SMILES.

# +
total_smilies = len(df)

invalids = df[df["SMILES_state"] == 0]
total_invalid_smilies = len(invalids)

duplicates = df[df["SMILES_state"] == 2]
total_batch_duplicate_smilies = len(duplicates)

all_duplicates = df[df.duplicated(subset=["SMILES"])]
total_duplicate_smilies = len(all_duplicates)

print(
    f"Total number of SMILES generated: {total_smilies}\n"
    f"Total number of invalid SMILES generated: {total_invalid_smilies}\n"
    f"Total number of batcg duplicate SMILES generated: {total_batch_duplicate_smilies}\n"
    f"Total number of duplicate SMILES generated: {total_duplicate_smilies}"
)
# -
# ### Display all generated duplicates

mol_view = create_mol_grid(all_duplicates)
display(mol_view)


# ### Display the molecules from the last step

last = df[df["step"] == max(df["step"])]
mol_view = create_mol_grid(last)
display(mol_view)

# ### Plot the NLLs
#
# The "Target" is the "augmented NLL".

# +
grouped_df = df.groupby("step")

for label in "Agent", "Prior", "Target":
    means = grouped_df.aggregate({label: "mean"})
    X = list(means.index.values)
    sns.scatterplot(means, x=X, y=label, label=label)

plt.xlabel("step")
plt.ylabel("NLL")
plt.show()
# -
