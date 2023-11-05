from reinvent.scoring.compute_scores import compute_component_scores
from reinvent_plugins.components.RDKit.comp_physchem import Qed


def test_scorer_cache():
    smilies = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N",
        "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)NC",
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    ]
    scoring_fct = Qed()
    mask = [1, 1, 1, 1, 1]

    cache = {}

    results = compute_component_scores(smilies, scoring_fct, cache, mask)

    assert len(cache) == len(smilies)
    assert len(results.scores) == 1  # 1 results
    assert len(results.scores[0]) == len(smilies)  # with N SMILES

    # 3 new SMILES
    smilies = [
        "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC",
    ]

    results = compute_component_scores(smilies, scoring_fct, cache, mask)

    assert len(cache) == 8
    assert len(results.scores[0]) == len(smilies)

    smilies = [
        "C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl",  # new
        "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC"  # new
        "CC(=O)Oc1ccccc1C(=O)O",  # old
        "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)NC",  # old
        "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC",  # old
    ]

    results = compute_component_scores(smilies, scoring_fct, cache, mask)

    assert len(cache) == 10
    assert len(results.scores[0]) == len(smilies)
