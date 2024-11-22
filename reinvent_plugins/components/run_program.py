"""Run an external subprocess"""

import subprocess as sp
from typing import List


def run_command(command: List[str], env: dict = None, input=None, cwd=None) -> sp.CompletedProcess:
    """Run an external command in a subprocess.

    :params command: array of command line arguments
    :returns: output object from the subprocess
    """

    args = dict(capture_output=True, text=True, check=True, shell=False)

    if env:
        args.update({"env": env})

    if input:
        args.update({"input": input})

    if cwd:
        args.update({"cwd": cwd})

    try:
        result = sp.run(command, **args)
    except sp.CalledProcessError as error:
        ret = error.returncode
        out = error.stdout
        err = error.stderr

        raise ValueError(
            f"{__name__}: {' '.join(command)} has failed with exit "
            f"code {ret}: stdout={out}, stderr={err}"
        )

    return result
