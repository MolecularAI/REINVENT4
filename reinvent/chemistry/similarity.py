import numpy as np
from rdkit.Chem import DataStructs


def calculate_tanimoto_batch(fp, fps) -> np.array:
    return np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))


def calculate_tanimoto(query_fps, ref_fingerprints) -> np.array:
    return np.array(
        [np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps]
    )


def calculate_dice_similarity_matrix(fps) -> np.array:
    n_fps = len(fps)

    sim = np.ones((n_fps, n_fps))
    for i in range(len(fps) - 1):
        sim[i, i + 1 :] = DataStructs.BulkDiceSimilarity(fps[i], fps[i + 1 :])
        sim.T[i, i + 1 :] = sim[i, i + 1 :]

    return sim


def calculate_tanimoto_similarity_matrix(fps) -> np.array:
    n_fps = len(fps)

    sim = np.ones((n_fps, n_fps))
    for i in range(len(fps) - 1):
        sim[i, i + 1 :] = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        sim.T[i, i + 1 :] = sim[i, i + 1 :]

    return sim
