"""Write out a CSV summary"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reinvent.runmodes.RL.reports import RLReportData

csv_logger = logging.getLogger("csv")
logger = logging.getLogger(__name__)

MODEL_SPECIFIC_HEADERS = {
    "Libinvent": [
        "Input_Scaffold",
        "R-groups",
    ],  # Named so to be different from Scaffold from diversity filter
    "Linkinvent": ["Warheads", "Linker"],
    "Mol2Mol": ["Input_SMILES"],
    "Pepinvent": ["Masked_input_peptide", "Fillers"],
}

FRAGMENT_GENERATORS = ["Libinvent", "Linkinvent", "Pepinvent"]


class RLCSVReporter:
    """CSV writer class"""

    def __init__(self, reporter):
        self.reporter = reporter
        self.__write_csv_header = True

    def submit(self, data: RLReportData) -> None:
        """Write a summary to the CSV logger

        :param data: data to be written
        :returns: headers and columns
        """

        header, columns = write_summary(data, write_header=self.__write_csv_header)
        self.__write_csv_header = False

        lines = [" | " + " ".join(header)]
        NUM_ROWS = 10  # FIXME

        for i, row in enumerate(zip(*columns)):
            if i >= NUM_ROWS:
                break

            out = []

            for item in row:
                if isinstance(item, (float, int, np.floating, np.integer)):
                    num = f"{item:.2f}"
                    out.append(num)
                elif item is None:
                    out.append("--")
                else:
                    out.append(item)

            lines.append(" | " + " ".join(out))

        lines = "\n".join(lines)

        # FIXME: wrong location?
        logger.info(
            f"Score: {data.mean_score:.2f} Agent NLL: {data.agent_mean_nll:.2f} "
            f"Valid: {round(100*data.fraction_valid_smiles):3d}% Step: {data.step}\n"
            f"{lines}"
        )


def write_summary(data, write_header):
    header = ["Agent", "Prior", "Target", "Score", "SMILES", "SMILES_state"]
    results = data.score_results

    columns = [
        [f"{score:.4f}" for score in data.agent_nll],
        [f"{score:.4f}" for score in data.prior_nll],
        [f"{score:.4f}" for score in data.augmented_nll],
        [f"{score:.7f}" for score in results.total_scores],
        results.smilies,
        [str(state.value) for state in data.sampled.states],
    ]

    if data.model_type in MODEL_SPECIFIC_HEADERS.keys():
        header.extend(MODEL_SPECIFIC_HEADERS[data.model_type])

        columns.append(data.sampled.items1)

        if data.model_type in FRAGMENT_GENERATORS:
            columns.append(data.sampled.items2)

    if data.scaffolds:
        header.append("Scaffold")
        columns.append(data.scaffolds)

    names = []
    scores = []
    raw_scores = []

    for transformed_result in results.completed_components:
        names.extend(transformed_result.component_names)

        for transformed_scores in transformed_result.transformed_scores:
            _scores = []

            for score in transformed_scores:
                try:
                    _scores.append(f"{float(score):.7f}")
                except ValueError:
                    _scores.append(score)

            scores.append(_scores)

        for original_scores in transformed_result.component_result.scores:
            _scores = []

            for score in original_scores:
                try:
                    _scores.append(f"{float(score):.4f}")
                except ValueError:
                    _scores.append(score)

            raw_scores.append(_scores)

    for name, score, raw_score in zip(names, scores, raw_scores):
        header.extend((name, f"{name} (raw)"))
        columns.extend((score, raw_score))

    if write_header:
        csv_logger.info(header + ["step"])

    for row in zip(*columns):
        csv_logger.info(row + (data.step,))

    return header, columns
