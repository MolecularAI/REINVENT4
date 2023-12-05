"""Map category names to float score"""

__all__ = ["ValueMapping"]

from dataclasses import dataclass
import logging

import numpy as np

from .transform import Transform


logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    type: str
    mapping: dict


class ValueMapping(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.mapping = params.mapping

    def __call__(self, values) -> np.ndarray:
        transformed = []
        missmatches = set()

        for value in values:
            if value in self.mapping:
                transformed.append(self.mapping[value])
            elif value == "0.0" or value == 0.0:
                transformed.append(float(value))
            else:
                transformed.append(np.nan)
                missmatches.add(value)

        if missmatches:
            len_miss = len(missmatches)

            logger.warning(
                f"The key{'s' if len_miss > 1 else ''} "
                f"'{', '.join([str(s) for s in list(missmatches)])}' have "
                "not been found in your provided mapping. The "
                "values have been substituted with NaNs."
            )

        return np.array(transformed, dtype=float)
