__all__ = ["ScaffoldPairGenerator"]
from functools import partial
import logging

import numpy as np
import pandas as pd

from reinvent.chemistry import conversions
from reinvent.chemistry.utils import compute_scaffold, compute_num_heavy_atoms
from reinvent.models.utils.parallel import parallel


from .pair_generator import PairGenerator


logger = logging.getLogger(__name__)


class ScaffoldPairGenerator(PairGenerator):
    """Generator of molecule pairs according to Scaffold similarity"""

    def __init__(self, generic: bool, add_same: bool = False, *args, **kwargs) -> None:
        """__init__.

        :param generic: whether to use the generic scaffold or not
        :type generic: bool
        :param add_same: whether to inlcude the pairs (s,s) or not
        :type add_same: bool
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        self.generic = generic
        self.add_same = add_same

    def build_pairs(self, smiles: list, *, processes: int = 8) -> pd.DataFrame:
        """build_pairs.

        :param smiles: a list containing smiles
        :type smiles: list
        :param processes: number of process for parallelizing the construction of pairs
        :type processes: int
        :rtype: pd.DataFrame
        """
        if len(smiles) == 0:
            raise ValueError("The smiles list is empty")

        logger.info(f"Creating Scaffold pairs with {processes:d} processes...")

        fsmiles, fscaffolds, fhatoms, fscaffold_hatoms = [], [], [], []

        for smi in smiles:
            mol = conversions.smile_to_mol(smi)
            scaffold = compute_scaffold(mol, generic=self.generic)
            hatoms = compute_num_heavy_atoms(mol)
            if (mol is not None) and (scaffold is not None):
                fsmiles.append(smi)
                fscaffolds.append(scaffold)
                fhatoms.append(hatoms)
                scaffold_mol = conversions.smile_to_mol(scaffold)
                fscaffold_hatoms.append(compute_num_heavy_atoms(scaffold_mol))
        fsmiles = np.array(fsmiles)
        fscaffolds = np.array(fscaffolds)
        fhatoms = np.array(fhatoms)
        fscaffold_hatoms = np.array(fscaffold_hatoms)
        del smiles

        shared_data = {
            "scaffold_db": fscaffolds,
            "hatom_db": fhatoms,
            "scaffold_hatom_db": fscaffold_hatoms,
            "smi_db": fsmiles,
        }

        parallel_build_pairs = partial(parallel, processes, shared_data, self._build_pairs)
        res = parallel_build_pairs(fscaffolds, fhatoms, fscaffold_hatoms, fsmiles)

        pd_cols = ["Source_Mol", "Target_Mol"]
        pd_data = []
        for r in res:
            pd_data = pd_data + r
        df = pd.DataFrame(pd_data, columns=pd_cols)
        df = df.drop_duplicates(subset=["Source_Mol", "Target_Mol"])
        df = self.filter(df)
        logger.info("Scaffold pairs created")
        return df

    def _build_pairs(
        self,
        scaffolds,
        hatoms,
        scaffold_hatoms,
        smiles,
        *,
        scaffold_db,
        hatom_db,
        scaffold_hatom_db,
        smi_db,
    ):
        table = []
        criterion_2 = scaffold_hatom_db >= hatom_db / 2.0
        for i in range(len(scaffolds)):
            criterion_3 = scaffold_hatoms[i] >= hatoms[i] / 2.0
            if criterion_3:
                criterion_1 = scaffolds[i] == scaffold_db
                idx = np.logical_and(criterion_2, criterion_1)
                for smi in smi_db[idx]:
                    if smiles[i] != smi:
                        table.append([smiles[i], smi])
                        table.append([smi, smiles[i]])
                    if self.add_same:
                        table.append([smi, smi])
                        table.append([smiles[i], smiles[i]])
        return table

    def get_params(self):
        params = {
            "generic": self.generic,
            "add_same": self.add_same,
        }

        return {**params, **super().get_params()}
