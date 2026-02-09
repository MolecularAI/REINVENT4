from argparse import ArgumentParser
from datetime import datetime
import os
import reinvent

def get_args():

    prior_filename = os.path.abspath(os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior"))
    now = datetime.now()
    date = now.strftime("%Y-%m-%d_%H-%M-%S")
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, default="both")
    parser.add_argument("--statistics", action='store_true')
    parser.add_argument("--wd", type=str, default=date)
    parser.add_argument("--prior", type=str, default=prior_filename)
    parser.add_argument("--s2", action='store_true')
    parser.add_argument("--tl-checkpoint", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--data-folder", type=str, default="dataset")
    parser.add_argument("--data-type", type=str, default="tack")

    return parser.parse_args()