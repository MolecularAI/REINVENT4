#!/bin/env python
#
# Installer for REINVENT4
#
# Simple script to compute the command line parameters for pip.  PyTorch
# controls the version for the processor type via setting the right URL. Some
# packages are optional and may need a URL of their own.
#

import subprocess as sp
import argparse

OPTIONAL_DEPENDENCIES = ("all", "none", "openeye", "isim")
OPENEYE_URL = "https://pypi.anaconda.org/OpenEye/simple"
PYTORCH_BASE_URL = "https://download.pytorch.org/whl"


def main(args):
    packages = args.optional_dependencies.split(",")

    extra_dependencies = None
    openeye_url = []

    if "none" in packages:
        extra_dependencies = ""
    elif "all" in packages:
        extra_dependencies = "[all]"
        openeye_url = ["--extra-index-url", OPENEYE_URL]
    else:
        extra_dependencies = "[" + ",".join(packages) + "]"

        if "openeye" in packages:
            openeye_url = ["--extra-index-url", OPENEYE_URL]

    if args.processor_type == "mac":
        pytorch_url = []
    else:
        pytorch_url = ["--extra-index-url", f"{PYTORCH_BASE_URL}/{args.processor_type}"]

    editable = None

    if args.editable:
        editable = "-e"

    cmd = ["pip", "install", editable, f".{extra_dependencies}"]
    cmd.extend(pytorch_url) 
    cmd.extend(openeye_url)
    final_cmd = list(filter(None, cmd))

    print(" ".join(final_cmd))
    sp.run(final_cmd)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"Simple installer for REINVENT4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "processor_type",
        default=None,
        metavar="NAME",
        help="PyTorch name of the processor type e.g. cu124, rocm6.2.4, cpu or mac",
    )

    parser.add_argument(
        "-d",
        "--optional-dependencies",
        metavar="PKGS",
        #       choices=OPTIONAL_DEPENDENCIES,
        default="all",
        help="Optional dependencies, comma separated, no spaces",
    )

    parser.add_argument(
        "-e",
        "--editable",
        action="store_true",
        help="editable install",
    )

    return parser.parse_args()


def main_script():
    """Main entry point from the command line"""

    args = parse_command_line()
    main(args)


if __name__ == "__main__":
    main_script()
