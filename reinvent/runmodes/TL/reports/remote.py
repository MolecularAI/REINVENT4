"""Send information to a remote server"""

from __future__ import annotations
from dataclasses import dataclass
from time import time
from typing import List

from reinvent.runmodes.reporter.remote import get_reporter

reporter = get_reporter()


@dataclass
class RemoteData:
    epoch: int
    epochs: int
    mean_nll: float
    mean_nll_valid: float = None


def send_report(data: RemoteData, reporter) -> None:
    """Send JSON data to a remote server

    :param data: data to be sent
    """

    if not reporter:
        return

    record = {
        "timestamp": time(),
        "epoch": data.epoch,
        "epochs": data.epochs,
        "mean_nll_train": data.mean_nll,
        "mean_nll_valid": data.mean_nll_valid,
    }
    reporter.send(record)
