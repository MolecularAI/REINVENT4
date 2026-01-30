#!/bin/bash



# TODO: Write python code to fix everything
WD_PATH=""
for arg in "$@"; do
    if [ "$arg" = "test" ]; then
        WD_PATH = $arg
        break
    fi
done
if [ -z "$WD_PATH" ]; then
    WD_PATH=$(python << EOF
import os
from datetime import datetime

now = datetime.now()
date = now.strftime("%Y-%m-%d_%H-%M-%S")
origin = f"{os.getcwd()}/runs"

if not os.path.isdir(origin):
    os.mkdir(origin)

print(f"{origin}/{date}")
EOF
    )
fi
echo $WD_PATH

#SBATCH --account=naiss2025-5-462
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=T4:1
#SBATCH --job-name=reinvent-test
#SBATCH --output=/mimer/NOBACKUP/groups/naiss2023-6-290/${USER}/REINVENT4_MasterThesis/${WD_PATH}/reinvent_test.log
#SBATCH --error=/mimer/NOBACKUP/groups/naiss2023-6-290/${USER}/REINVENT4_MasterThesis/${WD_PATH}/reinvent_test.err
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=${USER}@chalmers.se

# Load necessary modules
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
prj_dir=/mimer/NOBACKUP/groups/naiss2023-6-290/${USER}/REINVENT4_MasterThesis
# Source the virtual environment
source ${prj_dir}/reinvent4/bin/activate


srun python ${prj_dir}/notebooks/Reinvent_entrypoint.py "$@" --wd=$WD_PATH