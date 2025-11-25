#!/bin/bash
#
#SBATCH --job-name="synthsense"
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=cpu
################################################################################

export HITINVENT_PROFILES="stocks_and_reactions_profiles.json"

# Set up environment
export PATH="miniforge3/envs/reinvent4/bin:$PATH"
source "miniforge3/bin/activate" reinvent4

reinvent staged_learning.toml
