import numpy as np

from reinvent_plugins.components.RDKit.comp_mol_volume import Parameters, MolVolume


def test_comp_mol_volume():
    params = Parameters([0.2, 2.0])
    mol_volume = MolVolume(params)

    smiles = ["c1ccccc1N", "SCc1ccncc1O"]
    results = mol_volume(smiles)

    # EmbedMolecule now uses randomSeed=0xf00d, so output is deterministic
    # regardless of which tests ran before this one.
    expected = np.array([95.144, 123.712])
    assert np.allclose(np.concatenate(results.scores), expected, atol=1e-3)


def test_comp_mol_volume_none_molecule_scores_nan():
    # an invalid SMILES (None molecule from molcache) must score NaN rather
    # than raising (issue #333)
    params = Parameters([0.2])
    component = MolVolume(params)

    results = component(["not_a_molecule", "c1ccccc1N"])

    assert np.isnan(results.scores[0][0])
    assert not np.isnan(results.scores[0][1])
