import pytest
from types import SimpleNamespace
from dataclasses import dataclass

from reinvent.datapipeline.filters import RegexFilter


SMILES = [
"CC(=O)O[Cl+][O-]",
"CC(C)(C)[PH+]([BH2-][P+]([BH2-][ClH+2])(C(C)(C)C)C(C)(C)C)C(C)(C)C",
"[O-][Br+3]([O-])(O)Oc1cccnc1Br",
"CCC=CCI1C(=O)CCC1([IH])CC(=O)O",
"CC1=Cc2ccccc2S1(O[I+3]([O-])([O-])O)C(F)(F)F",
"C[N+]1(C2C[I-]C2)CC1"
]


@pytest.fixture
def regex_filter():
    config = SimpleNamespace(keep_stereo=True, keep_isotopes=False,
            max_heavy_atoms=70, max_mol_weight=1200, min_heavy_atoms=2,
            min_carbons=2, elements=["B", "P"])
    yield RegexFilter(config)
    pass  # no tear-down


def test_unwanted_tokens(regex_filter):
    good_smilies = []

    for smiles in SMILES:
        regex_smiles = regex_filter(smiles)
        print(regex_smiles)
        good_smilies.append(regex_smiles)

    assert len(good_smilies) == 6
    assert not all(good_smilies)
