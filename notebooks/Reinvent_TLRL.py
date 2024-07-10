# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Focus a _de novo_ model for Reinforcement Learning with the Reinvent prior
#
# This tutorial demonstrate how to focus a prior with transfer learning (RL) and use the new model for further reinforcement learning (RL).  We will use the Reinvent prior (_de novo_ model).
#
# We assume you run this tutorial from within the `notebook/` directory of the repository.

# +
import os
import shutil
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import reinvent
from reinvent.notebooks import load_tb_data, plot_scalars, get_image, create_mol_grid
from reinvent.scoring.transforms import ReverseSigmoid
from reinvent.scoring.transforms.sigmoids import Parameters as SigmoidParameters

import ipywidgets as widgets

# %load_ext tensorboard
# -

# ## Set up the first RL run
#
# Here, we will train a prior to generate more "drug-like" compounds as compared to the starting prior which was trained n ChEMBL data.
#
# This is essentially the same setup as in the Reinvent_demo notebook.

wd = "/tmp/R4_notebooks_output"
top = os.path.abspath(os.path.join(reinvent.__path__[0], ".."))
top

# ### Delete existing working directory and create a new one
#
# If the working directory already exists, it will be reused

# +
if not os.path.isdir(wd):
    shutil.rmtree(wd, ignore_errors=True)
    os.mkdir(wd)

os.chdir(wd)
wd
# -

# ### Write config file

# +
prior_filename = os.path.abspath(os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior"))
agent_filename = prior_filename

stage1_checkpoint = "stage1.chkpt"

stage1_parameters=f"""
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "tb_stage1"
json_out_config = "_stage1.json"

[parameters]

prior_file = "{prior_filename}"
agent_file = "{agent_filename}"
summary_csv_prefix = "stage1"

batch_size = 100

use_checkpoint = false

[learning_strategy]

type = "dap"
sigma = 128
rate = 0.0001

[[stage]]

max_score = 1.0
max_steps = 300

chkpt_file = "{stage1_checkpoint}"

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
stage1_config_filename = "stage1.toml"

with open(stage1_config_filename, "w") as tf:
    tf.write(stage1_parameters)
# -

# ##  Stage 1 Reinforcement Learning
#
# This run will take several minutes to finish and timings are provided at the end of the run.  After the run the TensorBoard output can be used to inspect the results of the run

shutil.rmtree("tb_stage1_0", ignore_errors=True)

# %%time
# !reinvent -l stage1.log $stage1_config_filename

# ### Inspect results with TensorBoard
#
# TensorBoard needs to be started after REINVENT has finished.

# %tensorboard --bind_all --logdir $wd/tb_stage1_0

# ## Transfer Learning to focus the model
#
# The aim of focussing a model is to train the model to generate molecules more like the input examples.  "More-like" here means that the loss is the mean NLL (negative log likelihood) of the molecule.  This means that training will lead to lower NLLs of sequences as in the input SMILES.

# ### Prepare the data
#
# We use the known Tankyrase-2 binders from [BindingDB](https://www.bindingdb.org/rwd/jsp/dbsearch/PrimarySearch_ki.jsp?tag=pol&submit=Search&target=tankyrase-2&polymerid=50006570,7858,9866).

bdb = pd.read_csv(f"{top}/notebooks/data/tnks2.csv")
bdb

# #### Clean data and extract "good" binders¶
#
# This is certainly a bit of a naive setup and is not meant to demonstrate the intricacies of data cleaning.  The literature should be consulted for more information e.g. [Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise](https://doi.org/10.1021/acs.jcim.4c00049).
#
# Here we simply filter by all compounds with an IC50 smaller than 1 μm and discard everything else.

clean = bdb[~bdb["exp (nM)"].str.match("[<>]")]
clean = clean.astype({'exp (nM)': 'float'})
len(bdb), len(clean)

good = clean[clean["exp (nM)"] < 1000]
good = good[good["exp_method"] != "EC50"]
good = good[good["exp_method"] != "Kd"]
good = good.rename(columns={"exp (nM)": "IC50"})
good = good.drop(columns=["exp_method"])

grid = create_mol_grid(good)
display(grid)

# #### Write the good binders to a SMILES file
#
# We will need this file for TL.  We also write the IC50 as comments but they will not be needed in TL
#
# We also split the SMILES into a training and a validation set.  Again, rather naivley, we do this randomlu.

# +
TL_train_filename = "tnks2_train.smi"
TL_validation_filename = "tnks2_validation.smi"

data = good.sample(frac=1)
n_head = int(0.8 * len(data))  # 80% of the data for training
n_tail = len(good) - n_head
print(f"number of molecules for: training={n_head}, validation={n_tail}")

train, validation = data.head(n_head), data.tail(n_tail)

train.to_csv(TL_train_filename, sep="\t", index=False, header=False)
validation.to_csv(TL_validation_filename, sep="\t", index=False, header=False)
# -

# #### TL setup

TL_parameters = f"""
run_type = "transfer_learning"
device = "cuda:0"
tb_logdir = "tb_TL"


[parameters]

num_epochs = 50
save_every_n_epochs = 2
batch_size = 100
sample_batch_size = 2000

input_model_file = "{stage1_checkpoint}"
output_model_file = "TL_reinvent.model"
smiles_file = "{TL_train_filename}"
validation_smiles_file = "{TL_validation_filename}"
standardize_smiles = true
randomize_smiles = true
randomize_all_smiles = false
internal_diversity = true
"""

# +
TL_config_filename = "transfer_learning.toml"

with open(TL_config_filename, "w") as tf:
    tf.write(TL_parameters)
# -

# ## Start Transfer Learning

shutil.rmtree("tb_TL", ignore_errors=True)

# !reinvent -l transfer_learning.log $TL_config_filename

# ### Inspect results with TensorBoard

# %tensorboard --bind_all --logdir $wd/tb_TL

# ### Choice of model
#
# The TL run has written out a checkpoint file every second step and now we will have to decide which checkpoint to use for RL.  This is really a judgment call for the user as TL in this context is not really a well defined problem with a well defined solution.  The aim of TL is to create a molecular distribution more like the molecules in the input SMILES but it is not principally clear how to quantify "more like".  On the one hand we do not want to stay too close to the original distribution and on the other hand we we do not want to create a model that, in the extrene, creates only molecules from the input distribution.
#
# So here we will use the model from step 30 whre the validation loss is minimal.  From the TensorBoard output we see that number of valid SMILES is slightly decreasing over the TL run but is still at 98% at step 40.  Duplicate SMILES generation initially decreases and increases again after around step 35 with a plateau of close-to-zero between steps 15 and 35.  Internal diversity is also decreasing over time but note from the y-axis that this is really very minimal.  **Please note, that when you run this example the results may be different and you may have to decide on a checkpoint file from a different step.**
#
# The TOML file for stage 2 will reuse most of the configuration from stage 1 as we will need to keep the original scoring functions active.  We only need to change the agent to the model file we have obtained from the TL run, increase the number of `max_steps`, and change filenames.

# +
TL_model_filename = os.path.join(wd, "TL_reinvent.model.30.chkpt")

stage2_parameters = re.sub("stage1", f"stage2", stage1_parameters)
stage2_parameters = re.sub("agent_file.*\n", f"agent_file = '{TL_model_filename}'\n", stage2_parameters)
stage2_parameters = re.sub("max_steps.*\n", f"max_steps = 500\n", stage2_parameters)
# -

# ## Stage 2 RL

# ### Predictive model (ChemProp)
#
# This is a [model](https://www.dropbox.com/scl/fi/zpnqc9at5a5dnkzfdbo6g/model.pt?rlkey=g005yli9364uptd94d60jtg5c&dl=0) that has been trained on free energy simulation data computed for the TNKS2 target.

chemprop_path = os.path.join(wd, "chemprop")

pred_model_parameters = f"""
[[stage.scoring.component]]
[stage.scoring.component.ChemProp]

[[stage.scoring.component.ChemProp.endpoint]]
name = "ChemProp"
weight = 0.6

params.checkpoint_dir = "{chemprop_path}"
params.rdkit_2d_normalized = true
params.target_column = "DG"

transform.type = "reverse_sigmoid"
transform.high = 0.0
transform.low = -50.0
transform.k = 0.4
"""


# ### Preview reverse sigmoid transform
#
# Plot the function to show how its parameters transform the input.

# +
def plot_transform(low, high, k):
    params = SigmoidParameters(type="reverse_sigmoid", high=high, low=low, k=k)
    reverse_sigmoid = ReverseSigmoid(params)
    x = np.linspace(low, high, num=25)
    vf = np.vectorize(reverse_sigmoid)
    
    plt.figure(figsize=(6, 3))
    ax = sns.lineplot(x=x, y=vf(x))
    ax.set(title="Reverse Sigmoid", xlabel="raw score", ylabel="transformed score")
    plt.show()

low = widgets.FloatSlider(min=-70, max=-30, step=5, value=-50.0)
high = widgets.FloatSlider(min=-20, max=20, step=5, value=0.0)
k = widgets.FloatSlider(min=0.1, max=0.7, step=0.1, value=0.4, orientation='vertical')

# +
p = widgets.interactive(plot_transform, low=low, high=high, k=k)

low_high_ctrl = widgets.HBox(p.children[:2], layout=widgets.Layout(flex_flow='row wrap'))
k_ctrl = p.children[2]
output = p.children[-1]
vbox = widgets.VBox([output, low_high_ctrl])

display(widgets.HBox([vbox, k_ctrl]))
# -

# If the widget above doesn't work, plot directly: change cell below to Code.

# + active=""
# plot_transform(-50.0, 0.0, 0.4)
# -

# ### Diversity Filter and Inception
#
# The Diversity Filter (DF) forces the agent to explore new scaffolds (here we use the Murcko scaffold decomposition algorithm from RDKit).  If the number count of the same scaffold exceeds 10 (`bucket_size`), all further occurences of the generated molecule containing that scaffold will be scored with zero.  This will only be enforced if the total score exceeds 0.7 (`minscore`) meaning that molecules lower than this score will not be considered for the DF filter.
#
# Inception is a form of replay memory.  This memory is used to compute part of the loss from a random sample from this memory (the other part is the augmented likelihood computed from the prior and agent likelihoods, and the current total score).  Here we chose a memory size of 50 (`memory_size`) and randomly sample 10 store molecules from it every step 10 steps (`sample_size`).  We could also seed  the memory with a set of SMILES of our own but please note that if those molecules do not score highly they will be removed from the memory very early in the run (we only store a finite size of molecules).

df_parameters = """
[diversity_filter]

type = "IdenticalMurckoScaffold"
bucket_size = 10
minscore = 0.7
"""

inception_parameters = """
[inception]

smiles_file = ""  # no seed SMILES
memory_size = 50
sample_size = 10
"""

# +
full_stage2_parameters = stage2_parameters + pred_model_parameters + df_parameters + inception_parameters 
stage2_config_filename = "stage2.toml"

with open(stage2_config_filename, "w") as tf:
    tf.write(full_stage2_parameters)
# -

# ### Run stage2
#
# This may take an hour or more.

# %%time
# !reinvent -l stage2.log $stage2_config_filename

# ### Inspect results with TensorBoard
#
# TensorBoard needs to be started after REINVENT has finished.

# %tensorboard --bind_all --logdir $wd/tb_stage2_0

# ## Discussion
#
# In this brief tutorial we can only capture a few key points.  In the TB output we see that all scoring functions are increasing in the run demonstrating that the agent learns to generate compounds more likely to follow the target (scoring) profile we have set up.  This is also confirmed by the loss functions (`Loss (likelihood averages)`) where we see prior and agent NLL drift apart and the agent NLL decreases over the run.  `QED` tells us that reasonable "drug-like" compounds (good quality molecules) are generated and in `ChemProp (raw)` we see that the compounds have increasinly better predicted binding affinity towards the TNKS2 target.  The percentage of valid SMILES is very high but we also produce many duplicates which may be a consequence of the very focussed model.
#
# Keep in mind that RL is very stochastic in nature and that a new run can produce rather different results.  It is therefore good practice to carry out multiple RL runs and aggregate the results to analyse the statistics of the run.
#
# Finally, we view some of the generated structures and choose to define "good binders" as molecules with a QED > 0.8 and a binding free energy ΔG < -25 kcal/mol.  The choice is a bit arbitrary but sensible.  Adjust to your own needs.  We see below that many of the predicited good binders have the same or similar scaffold as known binders.  In practice, we may want to analyse how many of the generated molecules are identical or very similar to those known binders.  This is a bit ambiguous though as rediscovery would demonstrate that the workflow works but on the other hand we are not interested in poducing already known results!  Further analysis could involve synthesisability estimation or ADMET models.  A typical workflow would require an elaborate post-processing step (filtering, clustering) to select candidate molecules to discuss with an actual project team and decide which compounds, if any, are useful for a given project. 

csv_file = os.path.join(wd, "stage2_1.csv")
df = pd.read_csv(csv_file)
df

# +
good_QED = df["QED"] > 0.8
good_dG = df["ChemProp (raw)"] < -25.0  # kcal/mol

good_binders = df[good_QED & good_dG]
len(good_binders)
# -

# #### Duplicate removal
#
# This can be easly done by finding duplicate SMILES.  The SMILES in the CSV file have been canonicalized.

good_binders = good_binders.drop_duplicates(subset=['SMILES'])
len(good_binders)

# ### Displaying good binders
#
# The display grid allows you to look at the scores and other data frome the datafram (the little "i" in the top right corner, click to make it sticky) and it is also possible to sort the compounds by these data.

grid = create_mol_grid(good_binders)
display(grid)


