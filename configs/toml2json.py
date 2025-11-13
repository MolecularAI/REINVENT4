#!/bin/env python3
#
# Simple script to convert a TOML file to a JSON file
#


import sys
import json

import tomli

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as tf:
        config = tomli.load(tf)

    with open(sys.argv[2], "w") as jf:
        json.dump(config, jf, indent=2)
