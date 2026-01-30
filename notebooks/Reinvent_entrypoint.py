from args import get_args
from run_RL import run_reinforcement_learning
from run_TL import run_transfer_learning
import os

def make_run_dirs(wd):

    # Create the folder for the run
    print("Working directory:", wd)
    if not os.path.isdir(wd):
        os.mkdir(wd)

    # Create the checkpoints out folder in the current run folder
    checkpoints_wd = f"{wd}/checkpoints"
    print("Working directory:", checkpoints_wd)
    if not os.path.isdir(checkpoints_wd):
        os.mkdir(checkpoints_wd)


if __name__ == "__main__":
    args = get_args()
    origin = f"{os.getcwd()}/runs"

    # Create the run folder if it does not exist
    if not os.path.isdir(origin):
        os.mkdir(origin)
    
    wd = f"{origin}/{args.wd}"
    make_run_dirs(wd)

    runform = str.lower(args.run)
    if runform== "tl":
        run_transfer_learning(args, wd)
    
    elif runform == "rl":
        run_reinforcement_learning(args, wd)

    elif runform == "both":
        run_transfer_learning(args, wd)
        run_reinforcement_learning(args, wd)

    
    else:
        raise Exception("The run method you tried to call is not implemented! Use one of: 'RL' 'TL' or 'Both'")




