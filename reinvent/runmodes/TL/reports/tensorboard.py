"""Write out a TensorBoard report"""

from __future__ import annotations

__all__ = ["TLTBReporter"]

from collections import Counter
from typing import TYPE_CHECKING

from reinvent.runmodes.utils import make_grid_image, compute_similarity_from_sample

if TYPE_CHECKING:
    from reinvent.runmodes.TL.reports import TLReportData

ROWS = 5
COLS = 6


class TLTBReporter:
    def __init__(self, reporter):
        self.reporter = reporter

    def submit(self, data: TLReportData) -> None:
        """Write out TensorBoard data

        :param data: data to be written out
        """

        smiles_counts = Counter(data.sampled_smilies)

        mean_nll_stats = {
            "Training Loss": data.mean_nll,
            "Sample Loss": data.sampled_nlls.mean(),
        }

        if data.mean_nll_validation is not None:
            mean_nll_stats["Validation Loss"] = data.mean_nll_validation

        self.reporter.add_scalars("A_Mean NLL loss", mean_nll_stats, data.epoch)

        self.reporter.add_scalar("B_Fraction valid SMILES", data.fraction_valid, data.epoch)
        self.reporter.add_scalar(
            "C_Fraction duplicate SMILES", data.fraction_duplicates, data.epoch
        )

        if data.internal_diversity > 0.0:
            self.reporter.add_scalar(
                "D_Internal Diversity of sample", data.internal_diversity, data.epoch
            )

        labels = [
            f"NLL={nll:.2f}({smiles_counts[smiles]})"
            for nll, smiles in zip(data.sampled_nlls, data.sampled_smilies)
        ]

        # FIXME: rows and cols depend on sample_batch_size
        image_tensor = make_grid_image(list(data.sampled_smilies), labels, ROWS * COLS, ROWS)

        if image_tensor is not None:
            self.reporter.add_image(
                f"Sampled structures",
                image_tensor,
                data.epoch,
                dataformats="CHW",
            )  # channel, height, width

        if data.reference_fingerprints:
            similarities = compute_similarity_from_sample(
                data.fingerprints, data.reference_fingerprints
            )
            self.reporter.add_histogram(
                "Tanimoto similarity on RDKitFingerprint", similarities, data.epoch
            )
