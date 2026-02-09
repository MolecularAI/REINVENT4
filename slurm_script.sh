#!/bin/bash
#SBATCH --account=naiss2025-5-462
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=protac_generation
#SBATCH --time=16:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexper@chalmers.se
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2023-6-290/%u/REINVENT4_MasterThesis/runs/slurm_job_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2023-6-290/%u/REINVENT4_MasterThesis/runs/slurm_job_%j.err

BASE_DIR=/mimer/NOBACKUP/groups/naiss2023-6-290/${USER}/REINVENT4_MasterThesis

WD_PATH=""
for arg in "$@"; do
    if [ "$arg" = "wd" ]; then
        WD_PATH=$arg
        break
    fi
done

if [ -z "$WD_PATH" ]; then
    WD_PATH=$(python -u << EOF
import os
from datetime import datetime

now = datetime.now()
date = now.strftime("%Y-%m-%d_%H-%M-%S")
origin = f"{os.getcwd()}/runs"

if not os.path.isdir(origin):
    os.mkdir(origin)

print(date)
EOF
    )
fi

python -u << EOF
import os
origin = f"{os.getcwd()}/runs"
if not os.path.isdir(f"{origin}/${WD_PATH}"):
    os.mkdir(f"{origin}/${WD_PATH}")
    os.utime(origin, None)
EOF

echo "Working directory: $WD_PATH"

# Load necessary modules
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
prj_dir=/mimer/NOBACKUP/groups/naiss2023-6-290/${USER}/REINVENT4_MasterThesis
# Source the virtual environment
source ${prj_dir}/reinvent4/bin/activate

# Run the job
mv ${BASE_DIR}/runs/slurm_job_${SLURM_JOB_ID}.log "${BASE_DIR}/runs/${WD_PATH}/"
mv ${BASE_DIR}/runs/slurm_job_${SLURM_JOB_ID}.err "${BASE_DIR}/runs/${WD_PATH}/"

srun -u python -u ${prj_dir}/notebooks/reinvent_entrypoint.py "$@" --wd=$WD_PATH
