"""
Patch for Tensorboard add_histogram
"""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._convert_np import make_np
from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto


def add_histogram(
    self,
    tag,
    values,
    global_step=None,
    bins="tensorflow",
    walltime=None,
    max_bins=None,
):
    torch._C._log_api_usage_once("tensorboard.logging.add_histogram")
    if self._check_caffe2_blob(values):
        from caffe2.python import workspace

        values = workspace.FetchBlob(values)
    if isinstance(bins, str) and bins == "tensorflow":
        bins = self.default_bins
    self._get_file_writer().add_summary(
        histogram(tag, values, bins, max_bins=max_bins), global_step, walltime
    )


def histogram(name, values, bins, max_bins=None):
    values = make_np(values)
    hist = make_histogram(values.astype(float), bins, max_bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram(values, bins, max_bins=None):
    """Convert values into a histogram proto using logic from histogram.cc."""
    if values.size == 0:
        raise ValueError("The input has no element.")
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    num_bins = len(counts)
    if max_bins is not None and num_bins > max_bins:
        subsampling = num_bins // max_bins
        subsampling_remainder = num_bins % subsampling
        if subsampling_remainder != 0:
            counts = np.pad(
                counts,
                pad_width=[[0, subsampling - subsampling_remainder]],
                mode="constant",
                constant_values=0,
            )
        counts = counts.reshape(-1, subsampling).sum(axis=-1)
        new_limits = np.empty((counts.size + 1,), limits.dtype)
        new_limits[:-1] = limits[:-1:subsampling]
        new_limits[-1] = limits[-1]
        limits = new_limits

    # Find the first and the last bin defining the support of the histogram:

    cum_counts = np.cumsum(counts > 0)  # this is the location of the bug
    start, end = np.searchsorted(cum_counts, [0, cum_counts[-1] - 1], side="right")
    start = int(start)
    end = int(end) + 1
    del cum_counts

    # TensorBoard only includes the right bin limits. To still have the leftmost limit
    # included, we include an empty bin left.
    # If start == 0, we need to add an empty one left, otherwise we can just include the bin left to the
    # first nonzero-count bin:
    counts = counts[start - 1 : end] if start > 0 else np.concatenate([[0], counts[:end]])
    limits = limits[start : end + 1]

    if counts.size == 0 or limits.size == 0:
        raise ValueError("The histogram is empty, please file a bug report.")

    sum_sq = values.dot(values)
    return HistogramProto(
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limit=limits.tolist(),
        bucket=counts.tolist(),
    )


__version = torch.__version__.split(".")
__major = int(__version[0])

if __major == 1:
    SummaryWriter.add_histogram = add_histogram
