from args import get_args
from run_rl import run_reinforcement_learning
from run_tl import run_transfer_learning
from prep_data import prep_data
import os

if __name__ == "__main__":
    args = get_args()

    runform = str.lower(args.run)

    if runform == "data":
        prep_data(args)

    else:
        origin = f"{os.getcwd()}/runs"

        # Create the run folder if it does not exist
        if not os.path.isdir(origin):
            os.mkdir(origin)
        
        wd = f"{origin}/{args.wd}"

        # Create the wd folder
        print("Working directory:", wd)
        if not os.path.isdir(wd):
            os.mkdir(wd)


        
        if runform== "tl":
            run_transfer_learning(args, wd, args.data_type)
        
        elif runform == "rl":
            run_reinforcement_learning(args, wd)

        elif runform == "both":
            
            os.mkdir(f"{wd}/Stage_1_RL")
            run_reinforcement_learning(args, f"{wd}/Stage_1_RL")

            os.mkdir(f"{wd}/Stage_2_TL")
            run_transfer_learning(args, f"{wd}/Stage_2_TL", "synthetic")

            os.mkdir(f"{wd}/Stage_3_RL")
            run_reinforcement_learning(args, f"{wd}/Stage_3_RL")

            os.mkdir(f"{wd}/Stage_4_TL")
            run_transfer_learning(args, f"{wd}/Stage_2_TL", "tack")

            os.mkdir(f"{wd}/Stage_5_RL")
            run_reinforcement_learning(args, f"{wd}/Stage_3_RL", True)

        elif runform == "rl_s2":
            run_reinforcement_learning(args, wd, True)

        elif runform == "data":
            prep_data(args)

        else:
            raise Exception("The run method you tried to call is not implemented! Use one of: 'RL' 'TL' or 'Both'")




