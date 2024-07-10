"""Send information to a remote server"""

from __future__ import annotations

__all__ = ["TLRemoteReporter"]
from collections import Counter
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reinvent.runmodes.TL.reports import TLReportData

logger = logging.getLogger(__name__)


class TLRemoteReporter:
    def __init__(self, reporter):
        self.reporter = reporter

    def submit(self, data: TLReportData) -> None:
        """Send JSON data to a remote server

        :param data: data to be sent
        """

        smiles_counts = Counter(data.sampled_smilies)

        record = {
            "epoch": data.epoch,
            "model_path": data.model_path,
            "learning_mean": {
                "training": data.mean_nll,
                "validation": data.mean_nll_validation,
                "sampled": float(data.sampled_nlls.mean()),
            },
            "smiles_report": [
                {"legend": f"Times sampled: {smiles_counts[smiles]:d}", "smiles": smiles}
                for smiles in smiles_counts
            ],
        }

        self.reporter.send(record)
