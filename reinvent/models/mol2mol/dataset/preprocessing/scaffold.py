__all__ = ["ScaffoldPairGenerator"]
from concurrent import futures

from tqdm import tqdm
import numpy as np
import pandas as pd

from reinvent.chemistry.conversions import Conversions
from reinvent.chemistry.utils import compute_scaffold, compute_num_heavy_atoms

from .pair_generator import PairGenerator


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

        # smiles = [s[0] for s in smiles]
        # data = self._standardize_smiles(smiles)
        data = smiles
        conversions = Conversions()
        fsmiles, fscaffolds, fhatoms, fscaffold_hatoms = [], [], [], []

        pbar = tqdm(data)
        pbar.set_description("Smiles to Chem.Mol")
        for smi in pbar:
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
        del data

        data_pool = []
        scaffold_chunks = np.array_split(fscaffolds, processes)
        hatom_chunks = np.array_split(fhatoms, processes)
        scaffold_hatoms_chunks = np.array_split(fscaffold_hatoms, processes)
        smile_chunks = np.array_split(fsmiles, processes)

        for pid, (sc_ck, ha_ck, sh_ck, smi_ck) in enumerate(
            zip(scaffold_chunks, hatom_chunks, scaffold_hatoms_chunks, smile_chunks)
        ):
            data_pool.append(
                {
                    "scaffolds": sc_ck,
                    "hatoms": ha_ck,
                    "scaffold_hatoms": sh_ck,
                    "smiles": smi_ck,
                    "scaffold_db": fscaffolds,
                    "hatom_db": fhatoms,
                    "scaffold_hatom_db": fscaffold_hatoms,
                    "smi_db": fsmiles,
                    "pid": pid,
                }
            )
        pool = futures.ProcessPoolExecutor(max_workers=processes)
        res = list(pool.map(self._build_pairs, data_pool))
        res = sorted(res, key=lambda x: x["pid"])

        pd_cols = ["Source_Mol", "Target_Mol"]
        pd_data = []
        for r in res:
            pd_data = pd_data + r["table"]
        df = pd.DataFrame(pd_data, columns=pd_cols)
        df = df.drop_duplicates(subset=["Source_Mol", "Target_Mol"])
        df = self.filter(df)
        return df

    def _build_pairs(self, args):
        scaffold_db = args["scaffold_db"]
        hatom_db = args["hatom_db"]
        scaffold_hatom_db = args["scaffold_hatom_db"]
        smi_db = args["smi_db"]
        scaffolds = args["scaffolds"]
        hatoms = args["hatoms"]
        scaffold_hatoms = args["scaffold_hatoms"]
        smiles = args["smiles"]
        pid = args["pid"]

        table = []
        criterion_2 = scaffold_hatom_db >= hatom_db / 2.0
        for i in tqdm(range(len(scaffolds))):
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
        return {"table": table, "pid": pid}
