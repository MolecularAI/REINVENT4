"""Send information to a remote server"""

from __future__ import annotations
from dataclasses import dataclass
import logging

from reinvent.runmodes.reporter.remote import get_reporter

reporter = get_reporter()
logger = logging.getLogger(__name__)


@dataclass
class RemoteData:
    epoch: int
    model_path: str
    mean_nll: float
    sampled_smiles: list[str]
    mean_nll_valid: float = None


def send_report(data: RemoteData, reporter) -> None:
    """Send JSON data to a remote server

    :param data: data to be sent
    """

    if not reporter:
        return

    smiles_counts = {}

    for smi in data.sampled_smiles:
        if smi not in smiles_counts:
            smiles_counts[smi] = 0
        smiles_counts[smi] += 1

    record = {
        "epoch": data.epoch,
        "model_path": data.model_path,
        "learning_mean": {
            "sampled": data.mean_nll_valid,  # this is actually validation
            "training": data.mean_nll,
        },
        "smiles_report": [
            {"legend": f"Times sampled: {smiles_counts[smi]:d}", "smiles": smi}
            for smi in smiles_counts
        ],
    }

    logger.debug(f"Remote reporter record:\n{record}")
    reporter.send(record)
