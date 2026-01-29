#!/bin/bash
#SBATCH --account=naiss2025-5-462
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=reinvent-test
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2023-6-290/alexper/REINVENT4_MasterThesis/slurm_out_dir/reinvent_test.log
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2023-6-290/alexper/REINVENT4_MasterThesis/slurm_out_dir/reinvent_test.err
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=alexper@chalmers.se

# Load necessary modules
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
prj_dir=/mimer/NOBACKUP/groups/naiss2023-6-290/alexper/REINVENT4_MasterThesis
# Source the virtual environment
source ${prj_dir}/reinvent4/bin/activate


srun python ${prj_dir}/notebooks/Reinvent_demotest.py