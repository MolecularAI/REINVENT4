import pytest

from reinvent.runmodes.RL.memories.inception import InceptionMemory


SMILIES = (
    "CCN(CC(=O)Nc1ccc(S(N)(=O)=O)cc1)CC(=O)Nc1ccccc1OC",
    "CC(c1nnnn1C)N(C)C1CCN(CCn2cccn2)CC1",
    "Nc1nc(NCCN2CCC(CO)CC2)cc(C2CC(N)C2)n1",
    "CCc1nc(CN2CCN(Cc3ccc4c(c3)OCO4)C(=O)C2)n[nH]1",
    "CN(C)CCNC(=O)Cn1cc(NC(=O)C2CCC2)cn1",
    "COc1cc(F)c(C(=O)NCCNC2CCCCCC2)cc1OC",
    "O=c1c2ccccc2n2c(SCc3cccc(Oc4ccccc4)c3)nnc2n1-c1ccccc1",
    "COC(=O)C(c1ccccc1Cl)N1CCN(C(=O)c2ccc(Cl)cc2)CC1",
    "Cc1nn(CC(=O)c2c(-c3ccccc3)[nH]c3ccccc23)c(C)c1[N+](=O)[O-]",
    "CC(C)N1CCN(c2ccc(CCNCCCN3CCCC3)cc2)CC1",
    "COCCn1cnnc1CCNC(=O)CN1CCCC(C)C1",
    "O=C(c1cnc[nH]1)N1CCC2(CC1)CN(CC1CCOC1)CCO2",
    "Cc1nn(C)c(C)c1C1CCCN1CC(O)CN1CCOCC1",
    "CN(C)CC(NC(=O)c1ccc(=O)n(Cc2ccccc2Cl)c1)c1ccccc1",
    "Cc1cc(F)ccc1CC(=O)NCC1(O)CCCNC1",
    "O=C(NCCN1CCOCC1)Nc1cccc(OCc2cccc(F)c2)c1",
    "COc1ccc(CNCC(=O)Nc2c(C)cccc2C(C)C)cc1",
    "Cc1ccccc1NC(=O)Cc1nc(-c2cc(C)n(CC3COc4ccccc4O3)c2C)cs1",
    "Nc1ncc(CNCCOc2ccccn2)cn1",
    "C1CN(CCNC2CSCCSC2)CCN1",
)


@pytest.fixture
def inception_memory():
    scores = [x / 20.0 for x in range(1, 21)]
    nlls = list(range(1, 21))

    memory = InceptionMemory(maxsize=len(SMILIES))
    memory.add(SMILIES, scores, nlls)

    yield memory
    pass  # no tear-down


def test_inception_required_argument():
    with pytest.raises(TypeError):
        memory = InceptionMemory()


def test_inception_add(inception_memory):
    assert len(inception_memory.storage) == len(SMILIES)
    assert inception_memory._smilies == set(SMILIES)

    smilies, _, _ = zip(*inception_memory.storage)

    assert list(smilies) == list(reversed(SMILIES))


def test_inception_sample(inception_memory):
    sample = inception_memory.sample(4)

    assert len(sample) == 3  # SMILES, scores, NLLs
    assert len(sample[0]) == 4
    assert len(sample[1]) == 4
    assert len(sample[2]) == 4


def test_inception_add_more(inception_memory):
    smilies = (
        "CC1CCCCC1NC(=O)C(=O)NCCCN1CCOCC1",
        "CC(C)C(N)C(=O)N1CCN(c2ccc(CN(C)C)cn2)CC1",
        "O=CI(CN1CCCC(C(=O)Nc2ccccc2)C1)NCC(=O)Nc1ccc(F)c(F)c1",
    )
    scores = (0.1732, 0.9871, 0.6570)
    nlls = (12.19, 5.13, 7.87)

    inception_memory.add(smilies, scores, nlls)

    assert len(inception_memory.storage) == len(SMILIES)

    smilies, scores, nlls = zip(*inception_memory.storage)

    assert smilies == (
        "C1CN(CCNC2CSCCSC2)CCN1",
        "CC(C)C(N)C(=O)N1CCN(c2ccc(CN(C)C)cn2)CC1",
        "Nc1ncc(CNCCOc2ccccn2)cn1",
        "Cc1ccccc1NC(=O)Cc1nc(-c2cc(C)n(CC3COc4ccccc4O3)c2C)cs1",
        "COc1ccc(CNCC(=O)Nc2c(C)cccc2C(C)C)cc1",
        "O=C(NCCN1CCOCC1)Nc1cccc(OCc2cccc(F)c2)c1",
        "Cc1cc(F)ccc1CC(=O)NCC1(O)CCCNC1",
        "CN(C)CC(NC(=O)c1ccc(=O)n(Cc2ccccc2Cl)c1)c1ccccc1",
        "O=CI(CN1CCCC(C(=O)Nc2ccccc2)C1)NCC(=O)Nc1ccc(F)c(F)c1",
        "Cc1nn(C)c(C)c1C1CCCN1CC(O)CN1CCOCC1",
        "O=C(c1cnc[nH]1)N1CCC2(CC1)CN(CC1CCOC1)CCO2",
        "COCCn1cnnc1CCNC(=O)CN1CCCC(C)C1",
        "CC(C)N1CCN(c2ccc(CCNCCCN3CCCC3)cc2)CC1",
        "Cc1nn(CC(=O)c2c(-c3ccccc3)[nH]c3ccccc23)c(C)c1[N+](=O)[O-]",
        "COC(=O)C(c1ccccc1Cl)N1CCN(C(=O)c2ccc(Cl)cc2)CC1",
        "O=c1c2ccccc2n2c(SCc3cccc(Oc4ccccc4)c3)nnc2n1-c1ccccc1",
        "COc1cc(F)c(C(=O)NCCNC2CCCCCC2)cc1OC",
        "CN(C)CCNC(=O)Cn1cc(NC(=O)C2CCC2)cn1",
        "CCc1nc(CN2CCN(Cc3ccc4c(c3)OCO4)C(=O)C2)n[nH]1",
        "CC1CCCCC1NC(=O)C(=O)NCCCN1CCOCC1",
    )
    assert scores == (
        1.0,
        0.9871,
        0.95,
        0.9,
        0.85,
        0.8,
        0.75,
        0.7,
        0.657,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.25,
        0.2,
        0.1732,
    )
    assert nlls == (20, 5.13, 19, 18, 17, 16, 15, 14, 7.87, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 12.19)
