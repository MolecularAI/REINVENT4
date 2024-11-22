"""A collection of constants for testing"""

SIMPLE_TOKENS = {
    t: i
    for i, t in enumerate(
        ["$", "^", "(", ")", "1", "2", "3", "=", "C", "F", "N", "O", "S", "c", "n"]
    )
}

PARACETAMOL = "CC(=O)NC1=CC=C(C=C1)O"
SCAFFOLD_SUZUKI = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*]"

INVALID = "INVALID"
NONSENSE = "C1CC(Br)CCC1[ClH]"
ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"
CELECOXIB = "O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N"
IBUPROFEN = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
ETHANE = "CC"
PROPANE = "CCC"
BUTANE = "CCCC"
PENTANE = "CCCCC"
HEXANE = "CCCCCC"
METAMIZOLE = "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
COCAINE = "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC"
BENZENE = "c1ccccc1"
TOLUENE = "c1ccccc1C"
ANILINE = "c1ccccc1N"
AMOXAPINE = "C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl"
GENTAMICIN = "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC"
METHOXYHYDRAZINE = "CONN"
HYDROPEROXYMETHANE = "COO"
SCAFFOLD_SUZUKI = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*]"
CELECOXIB_FRAGMENT = "*c1cc(C(F)(F)F)nn1-c1ccc(S(N)(=O)=O)cc1"
REACTION_SUZUKI = "[*;$(c2aaaaa2),$(c2aaaa2):1]-!@[*;$(c2aaaaa2),$(c2aaaa2):2]>>[*:1][*].[*:2][*]"
DECORATION_SUZUKI = "[*]c1ncncc1"
TWO_DECORATIONS_SUZUKI = "[*]c1ncncc1|[*]c1ncncc1"
TWO_DECORATIONS_ONE_SUZUKI = "[*]c1ncncc1|[*]C"
SCAFFOLD_NO_SUZUKI = "[*:0]Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)"
DECORATION_NO_SUZUKI = "[*]C"
CELECOXIB_SCAFFOLD = "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*:0]"
SCAFFOLD_TO_DECORATE = "[*]c1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*]"

WARHEAD_PAIR = "*C1CCCCC1|*C1CCCC(ON)C1"
WARHEAD_TRIPLE = "*N(C)C|*Cc1cncc(C#N)c1|*C[C@@H](O)CC(=O)O"
WARHEAD_QUADRUPLE = "*C[C@@H](O)CC(=O)O|*N(C)C|*c1ccccc1|*Cc1cncc(C#N)c1"
LINKER_TRIPLE = "[*]Cc1ccc(-c2cccc(COc3cc(O[*])c(CN[*])cc3Cl)c2C)c(C)c1-c1ccccc1"

SCAFFOLD_SINGLE_POINT = "[*]C(=O)N[C@@H](C)C(=O)C(C)C"
SCAFFOLD_DOUBLE_POINT = "[*]C(CCn1nnc2ccccc2c1=O)C(=O)[*]"
SCAFFOLD_TRIPLE_POINT = "[*]c1nc([*])nc(N)c1C(=O)C[C@@H](CNS(N)(=O)=O)C(=O)[*]"
SCAFFOLD_QUADRUPLE_POINT = "[*]C(=O)C1CC(O)CN1C(O)C([n+]1cc([*])n(-c2ccc([*])c(F)c2)n1)C(C)(C)[*]"
THREE_DECORATIONS = "[*]c1ncncc1|[*]c1ncncc1|[*]C"
FOUR_DECORATIONS = "[*]c1ncncc1|[*]c1ncncc1|[*]C|[*]C"

PEPINVENT_INPUT1 = "?|N[C@@H](CO)C(=O)|?|N[C@@H](Cc1ccc(O)cc1)C(=O)|N(C)[C@@H]([C@@H](C)O)C(=O)|N[C@H](Cc1c[nH]cn1)C(=O)|N[C@@H](CC(=O)N)C2(=O)"
PEPINVENT_INPUT2 = "?|?|?|N[C@@H](CCC(=O)O)C(=O)|N[C@@H]([C@@H](C)O)C(=O)|NCC(=O)|N[C@@H](CCC(=O)O)C(=O)|N[C@@H](Cc1ccccc1)C(=O)|N[C@@H](CC(C)C)C(=O)O"
PEPINVENT_INPUT3 = "?|NCC(=O)|N[C@@H](CC(=O)O)C(=O)|N[C@@H]([C@H]1C[C@H](OC(C)(C)O1)CO)C(=O)|N(C)[C@@H](CCC(=O)O)C(=O)|?|N(C)[C@@H](Cc1c[nH]c2ccccc12)C(=O)|?|N[C@@H](CCSC)C(=O)|?|N[C@@H](Cc1c[nH]cn1)C(=O)|N[C@@H](c1sc(S3)nc1c1ccc(F)cc1)C(=O)O"
PEPINVENT_OUTPUT1 = "N2[C@@H](CC(=O)N)C(=O)|N[C@@H](CNC(=O)N1CCC[C@@H]1[C@H](O)C(F)(F)F)C(=O)"
PEPINVENT_OUTPUT2 = "N[C@@H](Cc1ccccc1)C(=O)|N[C@@H]([C@@H](C)O)C(=O)|NCC(=O)|N(C)[C@@H](CC(C)C)C(=O)|N1[C@@H](CCC1)C(=O)"

IBUPROFEN_TOKENIZED = [
    "^",
    "C",
    "C",
    "(",
    "C",
    ")",
    "C",
    "c",
    "1",
    "c",
    "c",
    "c",
    "(",
    "c",
    "c",
    "1",
    ")",
    "[C@@H]",
    "(",
    "C",
    ")",
    "C",
    "(",
    "=",
    "O",
    ")",
    "O",
    "$",
]

# MMP
CHEMBL4440201 = "Cc1ccccc1CN1CCN(c2cn[nH]c(=O)c2Br)CC1=O"
CHEMBL4475970 = "O=C1CN(c2cn[nH]c(=O)c2Cl)CCN1Cc1ccccc1OC(F)(F)F"
CHEMBL4442703 = "Cc1ccccc1CN1CCN(c2cn[nH]c(=O)c2C(F)(F)F)CC1=O"
CHEMBL4469449 = "Cc1ccccc1CN1CCN(c2cn[nH]c(=O)c2N2CCOCC2)CC1=O"
CHEMBL4546342 = "Cc1ccccc1CN1CCN(c2cn[nH]c(=O)c2C(C)C)CC1=O"
CHEMBL4458766 = "Cn1ncc2cccc(CN3CCN(c4cn[nH]c(=O)c4Cl)CC3=O)c21"
CHEMBL4530872 = "O=C1CN(c2cn[nH]c(=O)c2Cl)CCN1Cc1ccccc1C(F)(F)F"
CHEMBL4475647 = "CS(=O)(=O)c1ccccc1CN1CCN(c2cn[nH]c(=O)c2Cl)CC1=O"
