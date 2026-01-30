from args import get_args
from run_rl import run_reinforcement_learning
from run_tl import run_transfer_learning
import os

if __name__ == "__main__":
    args = get_args()
    origin = f"{os.getcwd()}/runs"

    # Create the run folder if it does not exist
    if not os.path.isdir(origin):
        os.mkdir(origin)
    
    wd = f"{origin}/{args.wd}"

    # Create the wd folder
    print("Working directory:", wd)
    if not os.path.isdir(wd):
        os.mkdir(wd)


    runform = str.lower(args.run)
    if runform== "tl":
        run_transfer_learning(args, wd)
    
    elif runform == "rl":
        run_reinforcement_learning(args, wd)

    elif runform == "both":
        
        os.mkdir(f"{wd}/Stage_1_RL")
        run_reinforcement_learning(args, f"{wd}/Stage_1_RL")

        os.mkdir(f"{wd}/Stage_2_TL")
        run_transfer_learning(args, f"{wd}/Stage_2_TL")

        os.mkdir(f"{wd}/Stage_3_RL")
        run_reinforcement_learning(args, f"{wd}/Stage_3_RL", True)

    elif runform == "rl_s2":
        run_reinforcement_learning(args, wd, True)

    
    else:
        raise Exception("The run method you tried to call is not implemented! Use one of: 'RL' 'TL' or 'Both'")




