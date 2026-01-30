
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, default="TL")
    parser.add_argument("--statistics", action='store_true')

    return parser.parse_args()