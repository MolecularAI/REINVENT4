import numpy as np
from rdkit.Chem import DataStructs


def calculate_tanimoto_batch(fp, fps) -> np.array:
    return np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))


def calculate_tanimoto(query_fps, ref_fingerprints) -> np.array:
    return np.array(
        [np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps]
    )
