import pytest
from types import SimpleNamespace
from dataclasses import dataclass

from reinvent.datapipeline.filters import RDKitFilter


SMILES = [
"CN1CCC23c4c5ccc(OC(=O)c6ccc7ccc(C(=O)Oc8ccc9c%10c8OC8C(O)C=CC%11C(C9)N(C)CCC%108%11)cc7c6)c4OC2C(O)C=CC3C1C5",
"CC1(C)CCC(C)(C)c2cc(N3c4cc5c(cc4B4c6oc7ccc(C89CC%10CC%11CC(C8)C%119%10)cc7c6N(c6ccccc6)c6cccc3c64)C(C)(C)CCC5(C)C)ccc21",
"CC1C2C3C4C5CC6C7C(C)C8(C)C9(C)C%10(C)C%11(C)C(C)(C)C%12(C)C1(C)C21C32C43C65C78C93C2%10C%121%11",
"CC12CCC/C(=N\OCC(=O)O)C1C13c4c5c6c7c8c9c%10c%11c%12c%13c%14c%15c(c6c6c4c4c%16c%17c%18c(c%19c%20c1c5c8c1c%20c5c%19c8c%18c%18c%19c%17c%17c4c6c%15c4c%14c6c%12c%12c%14c%11c(c91)c5c%14c8c%18c%12c6c%19c4%17)C%1623)C7C%10%13",
]


@pytest.fixture
def rdkit_filter():
    config = SimpleNamespace(max_ring_size=7, max_num_rings=12,
            keep_stereo=False, report_errors=False)
    yield RDKitFilter(config)
    pass  # no tear-down


@pytest.mark.xfail
def test_large_percent(rdkit_filter):
    good_smilies = []

    for smiles in SMILES:
        rdkit_smiles = rdkit_filter(smiles)
        good_smilies.append(rdkit_smiles)

    assert len(good_smilies) == 0
