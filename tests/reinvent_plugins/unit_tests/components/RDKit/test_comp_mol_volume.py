import numpy as np

from reinvent_plugins.components.RDKit.comp_mol_volume import Parameters, MolVolume


def test_comp_mol_volume():
    params = Parameters([0.2, 2.0])
    mol_volume = MolVolume(params)

    smiles = ["c1ccccc1N", "SCc1ccncc1O"]
    results = mol_volume(smiles)

    expected_results = [np.array([95.144, 123.544])]
    expected_results_new = [np.array([95.144, 123.04])]  # RDKit's new conformer generator
    expected_results_rdkit_2025 = [np.array([94.944, 123.584])]

    assert (
        np.allclose(np.concatenate(results.scores), expected_results)
        or np.allclose(np.concatenate(results.scores), expected_results_new)
        or np.allclose(np.concatenate(results.scores), expected_results_rdkit_2025)
    )
