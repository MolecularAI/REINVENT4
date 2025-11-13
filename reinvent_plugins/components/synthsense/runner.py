import json
import logging
import os
import pathlib
import shlex
import shutil
import stat
import tempfile

import yaml

from reinvent_plugins.components.run_program import run_command
from reinvent_plugins.components.synthsense.aizynthfinder_config import (
    ensure_custom_stock_is_inchikey,
    prepare_config,
)
from reinvent_plugins.components.synthsense.parameters import ComponentLevelParameters

logger = logging.getLogger("reinvent")


def run_aizynth(smilies: list[str], params: ComponentLevelParameters, epoch: int) -> dict:
    """Run custom command that executes AiZynth.

    This method assumes that custom command is as following:
    - takes SMILES strings, one per line, from stdin
    - writes output JSON to stdout
    - expects config.yml in the current directory
    Custom command can do anything else:
    - can create any additional files in the current directory
    - can call slurm, call REST API, wait, block etc

    This script creates a temporary directory inside current directory,
    and executes the command in that directory.
    """

    config, cmd = prepare_config(params)

    # Temporary directory for aizynthfinder output.
    # We need an option to keep the directory for debugging,
    # but TemporaryDirectory got option delete=False only in Python 3.12.
    # Using mkdtemp allows manual control of directory cleanup.
    tmpdir = tempfile.mkdtemp(prefix=f"reinvent-aizynthfinder-{epoch}-", dir=os.getcwd())

    # By default tempdir can be read only by the creating user.
    # GUI runs Reinvent by a "service user". To inspect content by other users,
    # add permissions for group (GRP), and keep all for user (USR).
    permissions = (
        stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
    )
    os.chmod(tmpdir, permissions)

    ensure_custom_stock_is_inchikey(config, tmpdir)

    configpath = pathlib.Path(tmpdir) / "config.yml"
    with open(configpath, "wt") as f:
        yaml.dump(config, f)

    input = "\n".join(smilies)

    result = run_command(shlex.split(cmd), input=input, cwd=tmpdir)

    if result.returncode != 0:
        logger.warning(
            f"AiZynth process returned non-zero returncode ({result.returncode})."
            f" Stderr:\n{result.stderr}"
        )
    out = json.loads(result.stdout)

    # Temporary directory clenaup.
    is_debug = logger.level <= logging.DEBUG
    if not is_debug:
        shutil.rmtree(tmpdir)

    return out
