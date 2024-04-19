"""Write out a TensorBoard report"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import math
import numpy as np
import torch

from reinvent.models.model_factory.sample_batch import SampleBatch
from reinvent.runmodes.samplers.reports.report import report_setup
from reinvent.runmodes.utils import make_grid_image
from reinvent.runmodes.utils.decorators import extra_dependencies
from reinvent.runmodes.utils.plot import plot_histogram, plot_scatter

ROWS = 5
COLS = 20


@dataclass
class TBData:
    fraction_valid_smiles: float
    fraction_unique_molecules: float
    time: int
    additional_report: dict
    image_tensor: torch.Tensor


def setup_TBData(sampled: SampleBatch, time: int, **kwargs):
    fraction_valid_smiles, fraction_unique_molecules, time,  additional_report = \
        report_setup(sampled, time, **kwargs)

    counter = collections.Counter(sampled.smilies)
    top_sampled = counter.most_common(ROWS * COLS)
    top_sampled_smiles, top_sampled_freq = zip(*top_sampled)
    image_tensor, _ = make_grid_image(top_sampled_smiles, top_sampled_freq, "Times sampled", ROWS * COLS, ROWS)

    return TBData(fraction_valid_smiles,
                  fraction_unique_molecules,
                  time,
                  additional_report,
                  image_tensor
                  )

def write_report(reporter, data: TBData) -> None:
    """Write out TensorBoard data

    :param reporter: TB reporter for writing out the data
    :param data: data to be written out
    """
    reporter.add_text('Data', f'Valid SMILES: {data.fraction_valid_smiles}%  '
                              f'Unique Molecules: {data.fraction_unique_molecules}% ')
    reporter.add_text('Time', f"{str(data.time)}s")

    if data.image_tensor is not None:
        reporter.add_image(
            f"Most Frequent Molecules", data.image_tensor
        )

    # report histgram and scatter plot if matplotlib available
    _report_histgram(data, reporter)
    _report_scatter(data, reporter)


@extra_dependencies("matplotlib")
def _report_histgram(data: TBData, reporter):
    title = ''
    xlabel = ''
    for key, value in data.additional_report.items():
        if 'Tanimoto' in key:
            bins = np.arange(0, 11) * 0.1
            xlabel = 'Tanimoto'
        elif 'Output_likelihood' in key:
            bins = range(math.floor(min(value)), math.ceil(max(value)) + 2)
            xlabel = 'Output_likelihood'
        else:
            bins = 50

        if 'valid' in key:
            title = 'valid'
        elif 'unique' in key:
            title = 'unique'

        figure = plot_histogram(value, xlabel, bins, title=f'{len(value)} {title}')
        reporter.add_figure(key, figure)


@extra_dependencies("matplotlib")
def _report_scatter(data: TBData, reporter):
    xlabel, ylabel = 'Tanimoto', 'Output_likelihood'
    x_key, y_key = f'{xlabel}_unique', f'{ylabel}_unique'
    if x_key in data.additional_report.keys() and y_key in data.additional_report.keys():
        x, y = data.additional_report[x_key], data.additional_report[y_key]
        figure = plot_scatter(x, y, xlabel=xlabel, ylabel=ylabel, title=f'{len(x)} Unique')
        reporter.add_figure(f'{xlabel}_{ylabel}', figure)