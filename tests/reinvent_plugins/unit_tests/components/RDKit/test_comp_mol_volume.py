import numpy as np

from reinvent_plugins.components.RDKit.comp_mol_volume import Parameters, MolVolume


def test_comp_mol_volume():
    params = Parameters([0.2, 2.0])
    mol_volume = MolVolume(params)

    smiles = ["c1ccccc1N", "SCc1ccncc1O"]
    results = mol_volume(smiles)

    expected_results = [np.array([95.144, 123.544])]

    assert np.allclose(np.concatenate(results.scores), expected_results)
