from pathlib import Path
import pytest

import torch

from reinvent.runmodes.TL.run_transfer_learning import run_transfer_learning
from reinvent.runmodes.utils.helpers import set_torch_device


@pytest.fixture
def setup(tmp_path, json_config, pytestconfig):
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    output_model_file = tmp_path / "TL_libinvent.model"

    config = {
        "parameters": {
            "input_model_file": json_config["LIBINVENT_PRIOR_PATH"],
            "smiles_file": json_config["TL_LIBINVENT_SMILES_PATH"],
            "output_model_file": output_model_file,
            "save_every_n_epochs": 2,
            "batch_size": 64,
            "sample_batch_size": 10,
            "num_epochs": 2,
            "num_ref": 2
        }
    }

    return config


@pytest.mark.integration
def test_transfer_learning(setup, tmp_path, pytestconfig):
    config = setup
    device = pytestconfig.getoption("device")

    run_transfer_learning(config, device)

    checkpoint_files = list(Path(tmp_path).glob("*.chkpt"))

    assert len(checkpoint_files) == 1

    model = torch.load(config["parameters"]["output_model_file"])
    keys = list(model.keys())

    assert keys == ["model_type", "version", "model", "decorator"]

    assert model["model_type"] == "Libinvent"


SMILES_PAIRS="""\
[*]C#CC(=O)C12CC3C(C)CCC3C3(C=O)CC1C=C(C(C)C)C32C(=O)[*]	*c1ccccc1|*O	CC(C)C1=CC2CC3(C=O)C4CCC(C)C4CC2(C(=O)C#Cc2ccccc2)C13C(=O)O
[*]C#CC(=O)C12CC3C(CCC3[*])C3(C=O)CC1C=C(C(C)[*])C32C(=O)[*]	*c1ccccc1|*C|*C|*O	CC(C)C1=CC2CC3(C=O)C4CCC(C)C4CC2(C(=O)C#Cc2ccccc2)C13C(=O)O
[*]C#CC(=O)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C	*c1ccc(F)cc1	CC12CCC(O)CC1=CCC1C2CCC2(C)C(C(=O)C#Cc3ccc(F)cc3)CCC12
[*]C#CC(=O)c1cn(Cc2ccccc2)nn1	*c1ccc([N+](=O)[O-])cc1	O=C(C#Cc1ccc([N+](=O)[O-])cc1)c1cn(Cc2ccccc2)nn1
[*]C#CC(C=CCOc1ccc(OCC(=O)[*])c(C)c1)c1ccc(Br)cc1	*c1ccccn1|*O	Cc1cc(OCC=CC(C#Cc2ccccn2)c2ccc(Br)cc2)ccc1OCC(=O)O
[*]C#CC(CC(=O)[*])c1ccc(OCC2COc3ccc(O[*])cc3O2)cc1	*C|*O|*C(C)C	CC#CC(CC(=O)O)c1ccc(OCC2COc3ccc(OC(C)C)cc3O2)cc1
[*]C#CC(CC(=O)[*])c1ccc(OCc2ccc3nc(-c4ccc(C(C)C)cc4)nn3c2)cc1	*C|*O	CC#CC(CC(=O)O)c1ccc(OCc2ccc3nc(-c4ccc(C(C)C)cc4)nn3c2)cc1
[*]C#CC(Cc1nnn[nH]1)c1ccc(O[*])cc1	*C|*Cc1ccc2sc(Cl)c(-c3ccccc3C)c2c1	CC#CC(Cc1nnn[nH]1)c1ccc(OCc2ccc3sc(Cl)c(-c4ccccc4C)c3c2)cc1
[*]C#CC(O)(c1ccc(C#N)c([*])c1)c1cncn1C	*c1ccc(Cl)cc1|*c1cccc2ccccc12	Cn1cncc1C(O)(C#Cc1ccc(Cl)cc1)c1ccc(C#N)c(-c2cccc3ccccc23)c1
[*]C#CC(O[*])C(NS(=O)(=O)c1ccc(-c2ccc(OC)cc2)cc1)C(=O)O	*c1ccccc1|*Cc1ccccc1	COc1ccc(-c2ccc(S(=O)(=O)NC(C(=O)O)C(C#Cc3ccccc3)OCc3ccccc3)cc2)cc1
[*]C#CC(c1ccc(OCc2ccc3scc([*])c3c2)cc1)[*]	*C|*c1ccccc1C(F)(F)F|*Cc1nnn[nH]1	CC#CC(Cc1nnn[nH]1)c1ccc(OCc2ccc3scc(-c4ccccc4C(F)(F)F)c3c2)cc1
[*]C#CC1(C(F)(F)F)C(=O)Nc2ccc(F)cc2N1CC=C	*C1CC1	C=CCN1c2cc(F)ccc2NC(=O)C1(C#CC1CC1)C(F)(F)F
[*]C#CC1(O)CC2CCC(C1)N2[*]	*c1cccc(F)c1|*C(=O)OC	COC(=O)N1C2CCC1CC(O)(C#Cc1cccc(F)c1)C2
[*]C#CC1(O)CCC2C3CCc4cc(C(=O)Nc5cccnc5[*])ccc4C3C(O[*])CC21[*]	*C|*C|*Cc1cccc(Cl)c1|*C	CC#CC1(O)CCC2C3CCc4cc(C(=O)Nc5cccnc5C)ccc4C3C(OCc3cccc(Cl)c3)CC21C
[*]C#CC1(O)CCC2C3CCc4cc(O)ccc4C3C([*])CC21C	*C|*CC	CC#CC1(O)CCC2C3CCc4cc(O)ccc4C3C(CC)CC21C
[*]C#CC12C=C([*])C(=O)C=C1C1([*])C=C([*])C(=O)C(C)(C)C1CC2	*c1ncc[nH]1|*C#N|*C|*C#N	CC1(C)C(=O)C(C#N)=CC2(C)C3=CC(=O)C(C#N)=CC3(C#Cc3ncc[nH]3)CCC12
[*]C#CC1=NCC(=O)N(C)c2cc(OC)c(OC)cc21	*c1ccccc1	COc1cc2c(cc1OC)N(C)C(=O)CN=C2C#Cc1ccccc1
[*]C#CC1CN([*])CCN1c1ncc(C(O)(C(F)(F)F)C(F)(F)F)cn1	*C|*S(=O)(=O)c1ccc(N)nc1	CC#CC1CN(S(=O)(=O)c2ccc(N)nc2)CCN1c1ncc(C(O)(C(F)(F)F)C(F)(F)F)cn1
[*]C#CC1CN([*])CCN1c1ncc(C(O)(C(F)(F)F)C(F)(F)F)s1	*C|*S(=O)(=O)c1ccc(N)nc1	CC#CC1CN(S(=O)(=O)c2ccc(N)nc2)CCN1c1ncc(C(O)(C(F)(F)F)C(F)(F)F)s1
[*]C#CC=CCN(Cc1cccc2c(Br)csc12)[*]	*C(C)(C)C|*C	CN(CC=CC#CC(C)(C)C)Cc1cccc2c(Br)csc12
[*]C#CCC(C1CCC2C3C(O)C=C4CC(O)CCC4([*])C3CCC21C)[*]	*C|*C|*C	CC#CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C
[*]C#CCC(O)C1CCC(C2CCC(C(O)CC#CCCOc3ccc([*])cc3)O2)O1	*CCOc1ccc(CCCC)cc1|*CCCC	CCCCc1ccc(OCCC#CCC(O)C2CCC(C3CCC(C(O)CC#CCCOc4ccc(CCCC)cc4)O3)O2)cc1
[*]C#CCCN1CC=C(c2ccccc2)CC1	*c1ccc2[nH]ccc2c1	C(#Cc1ccc2[nH]ccc2c1)CCN1CC=C(c2ccccc2)CC1
[*]C#CCN(c1cc(OC)cc(OC)c1)c1ccc2ncc([*])nc2c1	*C1(O)CNC1|*c1cnn(C)c1	COc1cc(OC)cc(N(CC#CC2(O)CNC2)c2ccc3ncc(-c4cnn(C)c4)nc3c2)c1
[*]C#CCN(c1cc(OC)cc(OC)c1)c1ccc2ncc([*])nc2c1	*C1CCCCC1|*c1cnn(C)c1	COc1cc(OC)cc(N(CC#CC2CCCCC2)c2ccc3ncc(-c4cnn(C)c4)nc3c2)c1
[*]C#CCN(c1cc(OC)cc(OC)c1)c1ccc2ncc([*])nc2c1	*c1ccccn1|*c1cnn(C)c1	COc1cc(OC)cc(N(CC#Cc2ccccn2)c2ccc3ncc(-c4cnn(C)c4)nc3c2)c1
[*]C#CCN(c1ccc2ncc(-c3cnn(C)c3)nc2c1)[*]	*C1(O)CNC1|*c1cc(OC)cc(OC)c1	COc1cc(OC)cc(N(CC#CC2(O)CNC2)c2ccc3ncc(-c4cnn(C)c4)nc3c2)c1
[*]C#CCN(c1ccc2ncc(-c3cnn(C)c3)nc2c1)[*]	*c1cccnc1|*c1cc(OC)cc(OC)c1	COc1cc(OC)cc(N(CC#Cc2cccnc2)c2ccc3ncc(-c4cnn(C)c4)nc3c2)c1
[*]C#CCN1CCC(Cc2ccc(F)cc2)CC1	*c1cc2nc(O)[nH]c2cc1OC	COc1cc2[nH]c(O)nc2cc1C#CCN1CCC(Cc2ccc(F)cc2)CC1
[*]C#CCN1c2ccccc2C23CC(C(=O)OC)N(C(=O)NC4CCCCC4)C2=NC(C(=O)OC)=C(C(=O)[*])C13	*C|*OC	CC#CCN1c2ccccc2C23CC(C(=O)OC)N(C(=O)NC4CCCCC4)C2=NC(C(=O)OC)=C(C(=O)OC)C13
[*]C#CCN1c2ccccc2C23CC(C(=O)OC)N(S(=O)(=O)CCC[*])C2=NC(C(=O)[*])=C(C(=O)OC)C13	*C|*C|*OC	CC#CCN1c2ccccc2C23CC(C(=O)OC)N(S(=O)(=O)CCCC)C2=NC(C(=O)OC)=C(C(=O)OC)C13
[*]C#CCN1c2ccccc2C23CC(C(=O)OC)N(S(=O)(=O)c4ccc(C(F)(F)F)cc4)C2=NC(C(=O)[*])=C(C(=O)OC)C13	*C|*OC	CC#CCN1c2ccccc2C23CC(C(=O)OC)N(S(=O)(=O)c4ccc(C(F)(F)F)cc4)C2=NC(C(=O)OC)=C(C(=O)OC)C13
[*]C#CCN1c2ccccc2C23CC(C(=O)OC)N([*])C2=NC(C(=O)OC)=C(C(=O)OC)C13	*C|*C(=O)OCCC#C	C#CCCOC(=O)N1C2=NC(C(=O)OC)=C(C(=O)OC)C3N(CC#CC)c4ccccc4C23CC1C(=O)OC
[*]C#CCN1c2ccccc2C23CC(C(=O)[*])N(S(=O)(=O)c4ccc(C#N)cc4)C2=NC(C(=O)OC)=C(C(=O)[*])C13	*C|*OC|*OC	CC#CCN1c2ccccc2C23CC(C(=O)OC)N(S(=O)(=O)c4ccc(C#N)cc4)C2=NC(C(=O)OC)=C(C(=O)OC)C13
[*]C#CCNC(=O)c1cc(C)nn(C(C)c2ccc(F)c(F)c2)c1=O	*c1ccc2ncc3nc(C)n(C(C)C)c3c2c1	Cc1cc(C(=O)NCC#Cc2ccc3ncc4nc(C)n(C(C)C)c4c3c2)c(=O)n(C(C)c2ccc(F)c(F)c2)n1
[*]C#CCNC(=O)c1cccn(Cc2ccc(F)c(F)c2)c1=O	*c1ccc2ncc(NC3CCN(CC)CC3)cc2c1	CCN1CCC(Nc2cnc3ccc(C#CCNC(=O)c4cccn(Cc5ccc(F)c(F)c5)c4=O)cc3c2)CC1
[*]C#CCNC(=O)c1cccn(Cc2ccc(F)c(F)c2)c1=O	*c1ccc2nccc(NC3CC3)c2c1	O=C(NCC#Cc1ccc2nccc(NC3CC3)c2c1)c1cccn(Cc2ccc(F)c(F)c2)c1=O
[*]C#CCNC(=O)c1cncn(Cc2ccc(F)c(F)c2)c1=O	*c1ccc2ncc(O)cc2c1	O=C(NCC#Cc1ccc2ncc(O)cc2c1)c1cncn(Cc2ccc(F)c(F)c2)c1=O
[*]C#CCOC(=O)C12CCC(C(=C)C)C1C1CCC3C(C)(CCC4C(C)(CO)C(O)CCC43[*])C1(C)CC2	*C|*C	C=C(C)C1CCC2(C(=O)OCC#CC)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(CO)C5CCC43C)C12
[*]C#CCOC1(C)CC(C)C2=NCCN3C(=O)OC(C)(C(C[*])OC(=O)C(C)C(=O)C(C)C1OC1OC(C)CC(N(C)C)C1O)C3C2C	*c1cccnc1|*C	CCC1OC(=O)C(C)C(=O)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(OCC#Cc2cccnc2)CC(C)C2=NCCN3C(=O)OC1(C)C3C2C
[*]C#CCOC1(C)CC(C)C2=NCCN3C(=O)OC(C)(C(C[*])OC(=O)C(C)C(=O)C(C)C1OC1OC(C)CC(N(C)[*])C1O)C3C2C	*c1ccccc1|*C|*C	CCC1OC(=O)C(C)C(=O)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(OCC#Cc2ccccc2)CC(C)C2=NCCN3C(=O)OC1(C)C3C2C
[*]C#CCON=C1C([*])C2OC(=O)OC2(C)C(CC)OC(=O)C(C)C(=O)C(C)C(OC2OC(C)CC([*])C2O)C(C)(OC)CC1[*]	*c1cccnc1|*C|*N(C)C|*C	CCC1OC(=O)C(C)C(=O)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(OC)CC(C)C(=NOCC#Cc2cccnc2)C(C)C2OC(=O)OC12C
[*]C#CCOc1cc(-c2nc3c([nH]2)c(=O)n(CC[*])c(=O)n3CCC)on1	*c1cccc(C(F)(F)F)c1|*C	CCCn1c(=O)c2[nH]c(-c3cc(OCC#Cc4cccc(C(F)(F)F)c4)no3)nc2n(CCC)c1=O
[*]C#CCOc1cc(COc2ccc([*])cc2)ccc1Sc1ccc(OCC(=O)[*])c2c1CCCC2	*c1cccnc1|*C(F)(F)F|*O	O=C(O)COc1ccc(Sc2ccc(COc3ccc(C(F)(F)F)cc3)cc2OCC#Cc2cccnc2)c2c1CCCC2
[*]C#CCOc1ccc(S(=O)(=O)CC2(C(=O)[*])CCCN(C(=O)C(C)[*])C2)cc1	*C|*NO|*C	CC#CCOc1ccc(S(=O)(=O)CC2(C(=O)NO)CCCN(C(=O)C(C)C)C2)cc1
[*]C#CCOc1ccc(S(=O)(=O)CC2(C(=O)[*])CCCN([*])C2)cc1	*C|*NO|*S(=O)(=O)C(C)C	CC#CCOc1ccc(S(=O)(=O)CC2(C(=O)NO)CCCN(S(=O)(=O)C(C)C)C2)cc1
[*]C#CCOc1ccc(S(=O)(=O)N2CC(CC(=O)[*])SC(C)(C)C2C(=O)[*])cc1	*C|*N|*NO	CC#CCOc1ccc(S(=O)(=O)N2CC(CC(N)=O)SC(C)(C)C2C(=O)NO)cc1
[*]C#CCOc1ccc(S(=O)(=O)N2CCSC(C)(C)C2C(=O)[*])cc1	*CCCN|*NO	CC1(C)SCCN(S(=O)(=O)c2ccc(OCC#CCCCN)cc2)C1C(=O)NO
[*]C#CCOc1ccc(S(=O)(=O)NC(Cc2cn(C[*])c3ccccc23)C(=O)[*])cc1	*C|*CCCOc1ccccc1|*O	CC#CCOc1ccc(S(=O)(=O)NC(Cc2cn(CCCCOc3ccccc3)c3ccccc23)C(=O)O)cc1
[*]C#CCn1c(N2CCCC(N)C2)cc(=O)n(Cc2cn(CCC(=O)[*])nn2)c1=O	*C|*OC	CC#CCn1c(N2CCCC(N)C2)cc(=O)n(Cc2cn(CCC(=O)OC)nn2)c1=O
[*]C#CCn1c(N2CCCC(N)C2)nc2c1c(=O)n(CCCN1CCC(C(=O)[*])CC1)c(=O)n2C	*C|*OC(C)C	CC#CCn1c(N2CCCC(N)C2)nc2c1c(=O)n(CCCN1CCC(C(=O)OC(C)C)CC1)c(=O)n2C
[*]C#CCn1c(N2CCCC(N)C2)nc2c1c(=O)n(Cc1ccnc3ccccc13)c(=O)n2C	*C	CC#CCn1c(N2CCCC(N)C2)nc2c1c(=O)n(Cc1ccnc3ccccc13)c(=O)n2C
[*]C#CCn1c(N2CCCC([*])C2)nc2c1c(=O)n(CC(=O)c1ccccc1N[*])c(=O)n2C	*C|*N|*C(=O)C(C)C	CC#CCn1c(N2CCCC(N)C2)nc2c1c(=O)n(CC(=O)c1ccccc1NC(=O)C(C)C)c(=O)n2C
[*]C#CCn1c(N2CCCNCC2)nc2cnn(Cc3nc(C)c4ccccc4n3)c(=O)c21	*C	CC#CCn1c(N2CCCNCC2)nc2cnn(Cc3nc(C)c4ccccc4n3)c(=O)c21
[*]C#CCn1c([*])nc2c1C(=O)N(Cc1nc([*])c3ccccc3n1)C1=NCCCN12	*C|*N1CCCC(N)C1|*C	CC#CCn1c(N2CCCC(N)C2)nc2c1C(=O)N(Cc1nc(C)c3ccccc3n1)C1=NCCCN12
[*]C#CCn1c([*])nc2c1c(=O)n(Cc1ncc([*])c3ccccc13)c(=O)n2C	*C|*N1CCCC(N)C1|*C	CC#CCn1c(N2CCCC(N)C2)nc2c1c(=O)n(Cc1ncc(C)c3ccccc13)c(=O)n2C
[*]C#Cc1c(-c2cc(Cl)c(C(=O)N3CCC([*])C3)c(Cl)c2)ncnc1[*]	*c1ccc(N)nc1|*N(C)C|*CC	CCc1ncnc(-c2cc(Cl)c(C(=O)N3CCC(N(C)C)C3)c(Cl)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2cc(F)c(C(=O)[*])c(Cl)c2)ncnc1[*]	*c1ccc(N)nc1|*N1CCn2c(C)cnc2C1|*CC	CCc1ncnc(-c2cc(F)c(C(=O)N3CCn4c(C)cnc4C3)c(Cl)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2cc(OC)cc(OC)c2)n(C)c2ccc(-c3ccc(OC)cc3)cc12	*c1ccsc1	COc1ccc(-c2ccc3c(c2)c(C#Cc2ccsc2)c(-c2cc(OC)cc(OC)c2)n3C)cc1
[*]C#Cc1c(-c2cc(OC)cc(OC)c2)n(C)c2ccc([*])cc12	*c1ccsc1|*c1cc(OC)c(OC)c(OC)c1	COc1cc(OC)cc(-c2c(C#Cc3ccsc3)c3cc(-c4cc(OC)c(OC)c(OC)c4)ccc3n2C)c1
[*]C#Cc1c(-c2cc(OC)cc(OC)c2)oc(=O)c2cc(OC)c(OC)cc12	*CCCO	COc1cc(OC)cc(-c2oc(=O)c3cc(OC)c(OC)cc3c2C#CCCCO)c1
[*]C#Cc1c(-c2ccc(C(=O)N(C)C3CCS(=O)(=O)C3)c(F)c2)ncnc1[*]	*c1ccc(N)nc1|*CC	CCc1ncnc(-c2ccc(C(=O)N(C)C3CCS(=O)(=O)C3)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)N3CC4CN(C(=O)OC(C)(C)C)CC4C3)c(F)c2)ncnc1[*]	*c1ccc(N)nc1|*CC	CCc1ncnc(-c2ccc(C(=O)N3CC4CN(C(=O)OC(C)(C)C)CC4C3)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)N3CCC(N4CCN(CC)CC4)CC3)c(F)c2)ncnc1[*]	*c1ccc(N)nc1|*CC	CCc1ncnc(-c2ccc(C(=O)N3CCC(N4CCN(CC)CC4)CC3)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)N3CCC(O)C3)c(Cl)c2)ncnc1[*]	*c1ccc(N)nc1|*CC	CCc1ncnc(-c2ccc(C(=O)N3CCC(O)C3)c(Cl)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)N3CCC(OC)CC3)c(F)c2)ccnc1C[*]	*c1ccc(N)nc1|*C	CCc1nccc(-c2ccc(C(=O)N3CCC(OC)CC3)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)N3CCN([*])CC3)c(F)c2)ncnc1[*]	*c1ccc(N)nc1|*S(=O)(=O)CC|*CC	CCc1ncnc(-c2ccc(C(=O)N3CCN(S(=O)(=O)CC)CC3)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)N3CCN([*])CC3)c(OC)c2)ccnc1C[*]	*c1ccc(N)nc1|*CCC|*C	CCCN1CCN(C(=O)c2ccc(-c3ccnc(CC)c3C#Cc3ccc(N)nc3)cc2OC)CC1
[*]C#Cc1c(-c2ccc(C(=O)N3CCN([*])CC3)c([*])c2)ncnc1[*]	*c1ccc(N)nc1|*C(=O)OCC|*C|*CC	CCOC(=O)N1CCN(C(=O)c2ccc(-c3ncnc(CC)c3C#Cc3ccc(N)nc3)cc2C)CC1
[*]C#Cc1c(-c2ccc(C(=O)N3CCNC(=O)C3)c(OC)c2)ncnc1[*]	*c1ccc(N)nc1|*CC	CCc1ncnc(-c2ccc(C(=O)N3CCNC(=O)C3)c(OC)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)[*])c(C)c2)ncnc1[*]	*c1ccc(N)nc1|*N1CC(O)CO1|*CC	CCc1ncnc(-c2ccc(C(=O)N3CC(O)CO3)c(C)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)[*])c(F)c2)ncnc1[*]	*c1ccc(N)nc1|*N1CCN(C2CCS(=O)(=O)C2)CC1|*CC	CCc1ncnc(-c2ccc(C(=O)N3CCN(C4CCS(=O)(=O)C4)CC3)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)[*])c(F)c2)ncnc1[*]	*c1ccc(N)nc1|*NC|*CC	CCc1ncnc(-c2ccc(C(=O)NC)c(F)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(C(=O)[*])c([*])c2)ncnc1[*]	*c1ccc(N)nc1|*N1CCC(N2CCCCC2)CC1|*C|*CC	CCc1ncnc(-c2ccc(C(=O)N3CCC(N4CCCCC4)CC3)c(C)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(F)c(C(=O)N(C)CC3CCN([*])CC3)c2)ccnc1[*]	*c1ccc(N)nc1|*C|*C	Cc1nccc(-c2ccc(F)c(C(=O)N(C)CC3CCN(C)CC3)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(-c2ccc(OCC(=O)NCCC)cc2)oc2cc(O)c(C(=O)[*])cc12	*c1cccc(Cl)c1|*O	CCCNC(=O)COc1ccc(-c2oc3cc(O)c(C(=O)O)cc3c2C#Cc2cccc(Cl)c2)cc1
[*]C#Cc1c(-c2cnc3ccccc3c2)ncnc1[*]	*c1ccc(N)nc1|*C	Cc1ncnc(-c2cnc3ccccc3c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(C(=O)[*])noc1-c1cc(C(C)C)c(O)cc1O	*N1CCN(C(=O)c2ccc(C(F)(F)F)nc2)CC1|*NCC	CCNC(=O)c1noc(-c2cc(C(C)C)c(O)cc2O)c1C#CN1CCN(C(=O)c2ccc(C(F)(F)F)nc2)CC1
[*]C#Cc1c(C)ncnc1N1CCC(OC)CC1	*c1cnc(C)c(NS(=O)(=O)c2ccccc2)c1	COC1CCN(c2ncnc(C)c2C#Cc2cnc(C)c(NS(=O)(=O)c3ccccc3)c2)CC1
[*]C#Cc1c(CC)ncnc1-c1cc(Cl)c(C(=O)[*])c(Cl)c1	*c1ccc(N)nc1|*N1CCC(C(C)(C)O)CC1	CCc1ncnc(-c2cc(Cl)c(C(=O)N3CCC(C(C)(C)O)CC3)c(Cl)c2)c1C#Cc1ccc(N)nc1
[*]C#Cc1c(CC)ncnc1N1CCN([*])CC1	*c1cnc(C)c(NS(=O)(=O)c2ccc(F)cc2)c1|*C	CCc1ncnc(N2CCN(C)CC2)c1C#Cc1cnc(C)c(NS(=O)(=O)c2ccc(F)cc2)c1
[*]C#Cc1c(CC)ncnc1N1CCc2cc(OC)c(OC)cc2C1	*c1ccc(N)nc1	CCc1ncnc(N2CCc3cc(OC)c(OC)cc3C2)c1C#Cc1ccc(N)nc1
"""
