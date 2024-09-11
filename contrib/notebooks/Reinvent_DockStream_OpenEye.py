# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # `REINVENT 4.4`: Reinforcement Learning with DockStream and OpenEye (docking)
#
#
# This is a simple example of running `Reinvent` with only 1 score component (`DockStream`) using OpenEye applications. To execute this notebook, make sure you have cloned the `DockStream` repository from GitHub and installed the conda environment.
# You also need OpenEye applications installed with binaries on PATH.

# ## 1. Set up the paths
# _Please update only the USER EDITABLE SECTION such that it reflects your system's installation and execute it._

# +
import os
import json
import toml


# ===== USER EDITABLE SECTION =====
# Edit these paths if your DockStream installation is different
dockstream_path = os.path.expanduser("/path/to/DockStream")
dockstream_env = os.path.expanduser("/path/to/miniforge3/envs/DockStream")

# Edit these file names for your specific protein target
apo_protein_filename = "apo_7xn1.pdb"  # Change this to your apo protein file name
reference_ligand_filename = "tacrine_7xn1.pdb"  # Change this to your reference ligand file name

# Set environment variables for OE applications and OE license
os.environ['PATH'] = "/path/to/openeye/bin:" + os.environ.get('PATH', '')
os.environ['OE_LICENSE'] = "/path/to/oe_license.txt"

# Change this prefix for your specific protein target
output_prefix = "p7xn1"

# Update project folder path
project_dir = os.path.expanduser("/path/to/project/folder")
# ===== END OF USER EDITABLE SECTION =====


# Update paths for entry points
target_preparator = os.path.join(dockstream_path, "target_preparator.py")
docker = os.path.join(dockstream_path, "docker.py")

input_data_dir = os.path.join(project_dir, "input_data")
output_data_dir = os.path.join(project_dir, "output_data")
logs_dir = os.path.join(project_dir, "logs")
config_dir = os.path.join(project_dir, "configs")
lig_docked_dir = os.path.join(output_data_dir, "ligands_docked")
scores_dir = os.path.join(output_data_dir, "docking_scores")

# Create necessary directories
for directory in [input_data_dir, output_data_dir, logs_dir, config_dir, lig_docked_dir, scores_dir]:
    os.makedirs(directory, exist_ok=True)

# Update file paths
apo_protein_path = os.path.join(input_data_dir, apo_protein_filename)
reference_ligand_path = os.path.join(input_data_dir, reference_ligand_filename)

target_prep_path = os.path.join(config_dir, f"{output_prefix}_target_prep.json")
fixed_pdb_path = os.path.join(input_data_dir, f"{output_prefix}_fixed_target.pdb")
receptor_path = os.path.join(input_data_dir, f"{output_prefix}_receptor.oeb")
receptor_path_oedu = os.path.join(input_data_dir, f"{output_prefix}_receptor.oedu")
log_file_target_prep = os.path.join(logs_dir, f"{output_prefix}_target_prep.log")
log_file_docking = os.path.join(logs_dir, f"{output_prefix}_docking.log")
log_file_reinvent = os.path.join(logs_dir, f"{output_prefix}_reinvent.log")

docking_path = os.path.join(config_dir, f"{output_prefix}_docking.json")
ligands_docked_path = os.path.join(lig_docked_dir, f"{output_prefix}_ligands_docked.sdf")
ligands_scores_path = os.path.join(scores_dir, f"{output_prefix}_scores.csv")
# -

# ## 2. Set up the Target Preparation JSON Configuration
# Update the PDBFixer block if needed. Other cavity definition methods can be found at https://github.com/MolecularAI/DockStream/blob/c62e6abd919b5b54d144f5f792d40663c9a43a5b/examples/target_preparation/OpenEye_target_preparation.json.

# +
tp_dict = {
  "target_preparation":
  {
    "header": {                                   # general settings
      "environment": {
      },
      "logging": {                                # logging settings (e.g. which file to write to)
        "logfile": log_file_target_prep
      }
    },
    "input_path": apo_protein_path,                  # this should be an absolute path
    "fixer": {                                    # based on "PDBFixer"; tries to fix common problems with PDB files
      "enabled": True,
      "standardize": True,                        # enables standardization of residues
      "remove_heterogens": True,                  # remove hetero-entries
      "fix_missing_heavy_atoms": True,            # if possible, fix missing heavy atoms
      "fix_missing_hydrogens": True,              # add hydrogens, which are usually not present in PDB files
      "fix_missing_loops": False,                 # add missing loops; CAUTION: the result is usually not sufficient
      "add_water_box": False,                     # if you want to put the receptor into a box of water molecules
      "fixed_pdb_path": fixed_pdb_path            # if specified and not "None", the fixed PDB file will be stored here
    },
    "runs": [                                     # "runs" holds a list of backend runs; at least one is required
      {
        "backend": "OpenEye",                # one of the backends supported ("AutoDockVina", "OpenEye", ...)
        "output": {
          "receptor_path": receptor_path      # the generated receptor file will be saved to this location
        },
        "parameters": {},
        "cavity": {                               # there are different ways to specify the cavity; here, a reference
                                                  # ligand is used
          "method": "reference_ligand",
          "reference_ligand_path": reference_ligand_path,
          "reference_ligand_format": "pdb"
}}]}}

with open(target_prep_path, 'w') as f:
    json.dump(tp_dict, f, indent=2)
# -

# ## 3. Run Target Preparation

# !{dockstream_env}/bin/python {target_preparator} -conf {target_prep_path}

# ## 4. Create Receptor OEDU file
# New file format used by 2024+ OE apps.

# !oeb2dureceptor -in {receptor_path} -out {receptor_path_oedu}

# ## 5. Set up the Docking JSON Configuration
# Update docking run parameters if needed. Different OE scoring functions can be found at https://docs.eyesopen.com/toolkits/cpp/dockingtk/scoring.html.

# +
ed_dict = {
  "docking": {
    "header": {
      "environment": {
      },
      "logging": {
        "logfile": log_file_docking
      }
    },
    "ligand_preparation": {
      "embedding_pools": [
        {
          "pool_id": "Omega_pool",
          "type": "Omega",
          "parameters": {
            "mode": "classic"
          },
          "input": {
            "standardize_smiles": False,
            "type": "console"
          }
        }
      ]
    },
    "docking_runs": [
      {
        "backend": "Hybrid",
        "run_id": "Hybrid",
        "input_pools": [
          "Omega_pool"
        ],
        "parameters": {
          "seed": 42,
          "receptor_paths": [
            receptor_path_oedu
          ],
          "scoring": "Chemgauss4",
          "resolution": "High",
          "number_poses": 3
        },
        "output": {
          "poses": {
            "poses_path": ligands_docked_path,
            "overwrite": False
          },
          "scores": {
            "scores_path": ligands_scores_path,
            "overwrite": False
          }
        }
      }
    ]
  }
}

with open(docking_path, 'w+') as f:
    json.dump(ed_dict, f, indent=2)
# -

# ## 6. Set up Reinvent .toml configuration file

# +
toml_string = f"""
# REINVENT4 TOML input example for reinforcement/curriculum learning
#
#
# Curriculum learning in REINVENT4 is a multi-stage reinforcement learning
# run.  One or more stages (auto CL) can be defined.  But it is also
# possible to continue a run from any checkpoint file that is generated
# during the run (manual CL).  Currently checkpoints are written at the end
# of a run also when the run is forcefully terminated with Ctrl-C.


run_type = "staged_learning"
device = "cuda:0"  # Edit this if you want to use a different device
tb_logdir = "/tb_logs"  # Relative path to the TensorBoard logs directory  # Edit this path as needed
#json_out_config = "_staged_learning.json"  # write this TOML to JSON

[parameters]

# Uncomment one of the comment blocks below.  Each generator needs a model
# file and possibly a SMILES file with seed structures.  If the run is to
# be continued after termination, the agent_file would have to be replaced
# with the checkpoint file.


use_checkpoint = true  # if true read diversity filter from agent_file
purge_memories = false  # if true purge all diversity filter memories after each stage

## Reinvent
prior_file = "/path/to/reinvent.model.chkpt" # use checkpoint files or priors
agent_file = "/path/to/reinvent.model.chkpt"

## LibInvent
#prior_file = "priors/libinvent.prior"
#agent_file = "priors/libinvent.prior"
#smiles_file = "scaffolds.smi"  # 1 scaffold per line with attachment points

## LinkInvent
#prior_file = "priors/linkinvent.prior"
#agent_file = "priors/linkinvent.prior"
#smiles_file = "warheads.smi"  # 2 warheads per line separated with '|'

## Mol2Mol
#prior_file = "priors/mol2mol_scaffold_generic.prior"
#agent_file = "priors/mol2mol_scaffold_generic.prior"
#smiles_file = "mol2mol.smi"  # 1 compound per line
#sample_strategy = "multinomial"  # multinomial or beamsearch (deterministic)
#distance_threshold = 100

batch_size = 128          # network

unique_sequences = true  # if true remove all duplicates raw sequences in each step
                         # only here for backward compatibility
randomize_smiles = true  # if true shuffle atoms in SMILES randomly


[learning_strategy]

type = "dap"      # dap: only one supported
sigma = 128       # sigma of the RL reward function
rate = 0.0001     # for torch.optim


[diversity_filter]  # optional, comment section out or remove if unneeded
                    # NOTE: also memorizes all seen SMILES

type = "IdenticalMurckoScaffold" # IdenticalTopologicalScaffold,
                                 # ScaffoldSimilarity, PenalizeSameSmiles
bucket_size = 25                 # memory size in number of compounds
minscore = 0.4                   # only memorize if this threshold is exceeded
minsimilarity = 0.4              # minimum similarity for ScaffoldSimilarity
penalty_multiplier = 0.5         # penalty factor for PenalizeSameSmiles


# Reinvent only: guide RL in the initial phase
#[inception]  # optional, comment sectionout or remove if unneeded

#smiles_file = "sampled.smi"  # "good" SMILES for guidance
#memory_size = 100  # number of total SMILES held in memory
#sample_size = 10  # number of SMILES randomly chosen each epoch


### Stage 1
### Note that stages must always be a list i.e. double brackets
[[stage]]

chkpt_file = '/path/to/rl_run.chkpt'  # Edit this checkpoint file path
termination = "simple"  # termination criterion fot this stage
max_score = 0.6  # terminate if this total score is exceeded
min_steps = 25  # run for at least this number of steps
max_steps = 1000  # terminate entire run when exceeded

# Optionally, a DF can be set for each stage but note that the global DF
# section above will always overwrite the stage section and you need to
# delete [diversity_filter] to avoid this
#
#[stage.diversity_filter]
#type = "IdenticalMurckoScaffold"
# etc.

[stage.scoring]
type = "geometric_mean"  # aggregation function

[[stage.scoring.component]]
[[stage.scoring.component.DockStream.endpoint]]
name = "Docking"
weight = 1

params.configuration_path = {docking_path}
params.docker_script_path = {docker}
params.docker_python_path =  "{dockstream_env}/bin/python"
transform.type = "reverse_sigmoid"
transform.high = -8
transform.low = -16
transform.k = 0.25

### Stage 2
### next stage if wanted
# [[stage]]
# ...
"""

# Define the output path
toml_path = os.path.join(project_dir, "config.toml")

# Parse the TOML string
toml_dict = toml.loads(toml_string)

# Write the TOML content to a file
with open(toml_path, 'w') as f:
    toml.dump(toml_dict, f)
# -

# ## 7. Run REINVENT

# !reinvent -l {log_file_reinvent} {toml_path}

# ## 6. Analyse the Results
#
# To analyze the run using TensorBoard:
# 1. Open a terminal and navigate to the project directory.
# 2. Run the command: `tensorboard --logdir=tb_logs`
# 3. Open the provided URL (usually http://localhost:6006) in your web browser to view the TensorBoard dashboard.
#
# All docking scores are saved inside `summary_1.csv` file.
