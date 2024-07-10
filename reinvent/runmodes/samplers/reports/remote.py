"""Send information to a remote server"""

from __future__ import annotations

__all__ = ["SamplingRemoteReporter"]
import collections
from typing import TYPE_CHECKING

from reinvent.runmodes.samplers.reports.common import common_report

if TYPE_CHECKING:
    from reinvent.models.model_factory.sample_batch import SampleBatch

ROWS = 20
COLS = 5


class SamplingRemoteReporter:
    def __init__(self, reporter):
        self.reporter = reporter

    def submit(self, sampled: SampleBatch, **kwargs):
        fraction_valid_smiles, fraction_unique_molecules, additional_report = common_report(
            sampled, **kwargs
        )

        counter = collections.Counter(sampled.smilies)
        top_sampled = counter.most_common(ROWS * COLS)
        smiles_report = [{"smiles": smiles, "legend": legend} for smiles, legend in top_sampled]

        data = dict(
            fraction_valid_smiles=fraction_valid_smiles,
            fraction_unique_molecules=fraction_unique_molecules,
            smiles_report=smiles_report,
            additional_report=additional_report,
        )

        self.reporter.send(data)
