# NIBR substructure filter
#
# adapted from RDKit commit 4a69bc3493dd3e9bb9f7a519e306fbcb545f1452 which
#
# assign the filter

from collections import namedtuple

import numpy as np

from rdkit import Chem

FilterMatch = namedtuple(
    "FilterMatch",
    (
        "SubstructureMatches",
        "Min_N_O_filter",
        "Frac_N_O",
        "Covalent",
        "SpecialMol",
        "SeverityScore",
    ),
)


# Assign substructure filters and fraction of Nitrogen and Oxygen atoms
def assign_filters(catalog, mols):
    results = []

    inhouseFiltersCat = catalog

    NO_filter = "[#7,#8]"
    sma = Chem.MolFromSmarts(NO_filter, mergeHs=True)

    for mol in mols:
        qc, NO_filter, fracNO, co, sc, sm = [np.NaN] * 6

        try:
            # fraction of N and O atoms
            numHeavyAtoms = mol.GetNumHeavyAtoms()
            numNO = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7,#8]")))
            fracNO = float(numNO) / numHeavyAtoms

            # all substructure filters
            entries = inhouseFiltersCat.GetMatches(mol)

            if len(list(entries)):
                # initialize empty lists
                fs, sev, cov, spm = ([] for _ in range(4))

                # get the matches
                for entry in entries:
                    pname = entry.GetDescription()
                    n, s, c, m = pname.split("__")
                    fs.append(entry.GetProp("Scope") + "_" + n)
                    sev.append(int(s))
                    cov.append(int(c))
                    spm.append(int(m))

                # concatenate all matching filters
                qc = " | ".join(fs)

                # assign overall severity
                if sev.count(2):
                    sc = 10
                else:
                    sc = sum(sev)

                # get number of covalent flags and special molecule flags
                co = sum(cov)
                sm = sum(spm)

            # if non of the filters matches
            else:
                qc = "no match"
                sc = 0
                co = 0
                sm = 0

            # special NO filter
            if not mol.HasSubstructMatch(sma):
                NO_filter = "no_oxygen_or_nitrogen"
            else:
                NO_filter = "no match"
        except Exception:
            print("Failed on compound {0}\n".format(smi))
            pass

        results.append(FilterMatch(qc, NO_filter, fracNO, co, sm, sc))

    return results
