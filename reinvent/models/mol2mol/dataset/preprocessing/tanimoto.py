__all__ = ["TanimotoPairGenerator"]
from concurrent import futures

from tqdm import tqdm
import numpy as np
import pandas as pd
from reinvent.chemistry.conversions import Conversions
from reinvent.chemistry.similarity import Similarity

from .pair_generator import PairGenerator


class TanimotoPairGenerator(PairGenerator):
    """Generator of molecule pairs according to Tanimoto similarity"""

    def __init__(
        self,
        lower_threshold: float,
        upper_threshold: float = 1.0,
        add_same: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """__init__.

        :param lower_threshold: keeps all the pairs such that tanimoto(s,t) >= lower_threshold
        :type lower_threshold: float
        :param upper_threshold: keeps all the pairs such that tanimoto(s,t) <= upper_threshold
        :type upper_threshold: float
        :param add_same: whether to inlcude the pairs (s,s) or not
        :type add_same: bool
        :rtype: None
        """
        super().__init__(*args, **kwargs)
        if (lower_threshold > 1.0) or (lower_threshold < 0):
            raise ValueError("`lower_threshold` must be in [0,1]")
        if (upper_threshold > 1.0) or (upper_threshold < 0):
            raise ValueError("`upper_threshold` must be in [0,1]")
        if lower_threshold > upper_threshold:
            lower_threshold, upper_threshold = upper_threshold, lower_threshold

        self.lth = lower_threshold
        self.uth = upper_threshold
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
        lth = self.lth
        uth = self.uth

        data = smiles

        # FIXME: this should really not be here
        # data = self._standardize_smiles(smiles)

        conversions = Conversions()
        fsmiles, fmolecules = [], []

        pbar = tqdm(data)
        pbar.set_description("Smiles to Fingerprints")
        for smi in pbar:
            # FIXME: there are multiple ways to generate
            #        Morgan fingerprints. Here, the one
            #        used to train the prior based on PubChem.
            #        Those parameters should probably be saved
            #        in the model checkpoint or passed 
            #        in the configuration file.
            mol = conversions.smiles_to_fingerprints(
                [smi], radius=2, use_counts=True, use_features=False
            )
            if len(mol):
                fmolecules.append(mol[0])
                fsmiles.append(smi)
        fsmiles = np.array(fsmiles)
        del data

        data_pool = []
        mol_chunks = np.array_split(fmolecules, processes)
        smile_chunks = np.array_split(fsmiles, processes)
        for pid, (mchunk, schunk) in enumerate(zip(mol_chunks, smile_chunks)):
            data_pool.append(
                {
                    "molecules": mchunk,
                    "smiles": schunk,
                    "mol_db": fmolecules,
                    "smi_db": fsmiles,
                    "lth": lth,
                    "uth": uth,
                    "pid": pid,
                }
            )
        pool = futures.ProcessPoolExecutor(max_workers=processes)
        res = list(pool.map(self._build_pairs, data_pool))
        res = sorted(res, key=lambda x: x["pid"])

        pd_cols = ["Source_Mol", "Target_Mol", "Tanimoto"]
        pd_data = []
        for r in res:
            pd_data = pd_data + r["table"]
        df = pd.DataFrame(pd_data, columns=pd_cols)
        df = df.drop_duplicates(subset=["Source_Mol", "Target_Mol"])
        df = self.filter(df)
        return df

    def _build_pairs(self, args):
        mol_db = args["mol_db"]
        smi_db = args["smi_db"]
        mols = args["molecules"]
        smiles = args["smiles"]
        lth = args["lth"]
        uth = args["uth"]
        pid = args["pid"]
        similarity = Similarity()

        table = []

        for i, mol in tqdm(enumerate(mols), total=len(mols)):
            ts = similarity.calculate_tanimoto_batch(mol, mol_db)
            idx = (ts >= lth) & (ts <= uth)
            for smi, t in zip(smi_db[idx], ts[idx]):
                table.append([smiles[i], smi, t])
                table.append([smi, smiles[i], t])
                if self.add_same:
                    table.append([smi, smi, 1.0])
                    table.append([smiles[i], smiles[i], 1.0])
        return {"table": table, "pid": pid}
