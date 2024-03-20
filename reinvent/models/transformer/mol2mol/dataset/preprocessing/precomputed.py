__all__ = ["PrecomputedPairGenerator"]

import pandas as pd

from .pair_generator import PairGenerator


class PrecomputedPairGenerator(PairGenerator):
    """Generator of molecule pairs according to Tanimoto similarity"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_pairs(self, smiles: list, *, processes: int = 8) -> pd.DataFrame:
        """build_pairs.

        :param smiles: a list containing pairs of smiles
        :type smiles: list
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        :rtype: pd.DataFrame
        """
        if len(smiles) == 0:
            raise ValueError("The smiles list is empty")
        pd_cols = ["Source_Mol", "Target_Mol"]
        df = pd.DataFrame(smiles, columns=pd_cols)
        df = self.filter(df)
        return df

    def get_params(self):
        return super().get_params()
