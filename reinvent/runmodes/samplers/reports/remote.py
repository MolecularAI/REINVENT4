"""Send information to a remote server"""

from __future__ import annotations

import collections
from dataclasses import dataclass, asdict

from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.runmodes.reporter.remote import get_reporter
from reinvent.runmodes.samplers.reports.report import report_setup

ROWS = 20
COLS = 5


@dataclass
class RemoteData:
    fraction_valid_smiles: float
    fraction_unique_molecules: float
    time: int
    additional_report: dict
    smiles_report: dict

def setup_RemoteData(sampled: SampleBatch, time: int, **kwargs):
    fraction_valid_smiles, fraction_unique_molecules, time,  additional_report = \
        report_setup(sampled, time, **kwargs)

    counter = collections.Counter(sampled.smilies)
    top_sampled = counter.most_common(ROWS * COLS)
    smiles_report = [{"smiles": smiles, "legend": legend} for smiles, legend in top_sampled]

    return RemoteData(fraction_valid_smiles,
                      fraction_unique_molecules,
                      time,
                      additional_report,
                      smiles_report
                      )


def send_report(data: RemoteData) -> None:
    """Send data to a remote endpoint

    :param data: data to be send and transformed into JSON format
    """
    reporter = get_reporter()
    if not reporter:
        return
    record = dict(asdict(data))
    reporter.send(record)
