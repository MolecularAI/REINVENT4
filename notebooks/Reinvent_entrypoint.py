from args import get_args
from run_RL import run_reinforcement_learning
from run_TL import run_transfer_learning

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
import subprocess
from datetime import datetime





if __name__ == "__main__":
    args = get_args()

    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    wd = f"{os.getcwd()}/runs"
    if not os.path.isdir(wd):
        os.mkdir(wd)

    print("Working directory:", wd)

    runform = str.lower(args.run)
    if runform== "tl":
        wd += f"/TL_{datetime_string}"
        run_transfer_learning(args, wd)
    
    elif runform == "rl":
        wd += f"/RL_{datetime_string}"
        run_reinforcement_learning(args)

    elif runform == "both":
        wd += f"/TLRL_{datetime_string}"
        run_transfer_learning(args, wd)
        run_reinforcement_learning(args, wd)

    
    else:
        raise Exception("The run method you tried to call is not implemented! Use one of: 'RL' 'TL' or 'Both'")




