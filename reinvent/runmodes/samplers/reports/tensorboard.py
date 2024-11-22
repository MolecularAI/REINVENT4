"""Write out a TensorBoard report"""

from __future__ import annotations

__all__ = ["SamplingTBReporter"]
from collections import Counter
from typing import TYPE_CHECKING

import math
import numpy as np

from reinvent.runmodes.samplers.reports.common import common_report
from reinvent.runmodes.utils import make_grid_image
from reinvent.runmodes.utils.decorators import extra_dependencies
from reinvent.runmodes.utils.plot import plot_histogram, plot_scatter

if TYPE_CHECKING:
    from reinvent.models.model_factory.sample_batch import SampleBatch

ROWS = 5
COLS = 20


class SamplingTBReporter:
    def __init__(self, reporter):
        self.reporter = reporter

    def submit(self, sampled: SampleBatch, **kwargs):
        """Write out TensorBoard data

        :param sampled: data to be written out
        """

        fraction_valid_smiles, fraction_unique_molecules, additional_report = common_report(
            sampled, **kwargs
        )

        smiles_count = Counter(sampled.smilies)
        top_sampled = smiles_count.most_common(ROWS * COLS)
        top_sampled_smiles, top_sampled_freq = zip(*top_sampled)
        labels = [f"sampled={freq}x" for freq in top_sampled_freq]

        image_tensor = make_grid_image(top_sampled_smiles, labels, ROWS * COLS, ROWS)

        self.reporter.add_text(
            "Data",
            f"Valid SMILES fraction: {fraction_valid_smiles}  "
            f"Unique Molecules fraction: {fraction_unique_molecules} ",
        )

        if image_tensor is not None:
            self.reporter.add_image(f"Most Frequent Molecules", image_tensor)

        # report histgram and scatter plot if matplotlib available
        if additional_report:
            key, figure = report_histgram(additional_report)
            self.reporter.add_figure(key, figure)

            xlabel, ylabel = "Tanimoto", "Output_likelihood"

            if figure := report_scatter(additional_report, xlabel, ylabel):
                self.reporter.add_figure(f"{xlabel}_{ylabel}", figure)


@extra_dependencies("matplotlib")
def report_histgram(additional_report):
    title = ""
    xlabel = ""

    for key, value in additional_report.items():
        if "Tanimoto" in key:
            bins = np.arange(0, 11) * 0.1
            xlabel = "Tanimoto"
        elif "Output_likelihood" in key:
            bins = range(math.floor(min(value)), math.ceil(max(value)) + 2)
            xlabel = "Output_likelihood"
        else:
            bins = 50

        if "valid" in key:
            title = "valid"
        elif "unique" in key:
            title = "unique"

        figure = plot_histogram(value, xlabel, bins, title=f"{len(value)} {title}")

        return key, figure


@extra_dependencies("matplotlib")
def report_scatter(additional_report, xlabel, ylabel):
    x_key, y_key = f"{xlabel}_unique", f"{ylabel}_unique"

    if x_key in additional_report.keys() and y_key in additional_report.keys():
        x, y = additional_report[x_key], additional_report[y_key]

        return plot_scatter(x, y, xlabel=xlabel, ylabel=ylabel, title=f"{len(x)} Unique")
