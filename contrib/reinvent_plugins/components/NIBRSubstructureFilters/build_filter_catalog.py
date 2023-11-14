# NIBR substructure filter
#
# adapted from RDKit commit 4a69bc3493dd3e9bb9f7a519e306fbcb545f1452 which
# includes the CSV file SubstructureFilter_HitTriaging_wPubChemExamples.csv
#
# build the filter from a CSV filer and write it out in pickle format

from collections import namedtuple

import pandas as pd

from rdkit.Chem import FilterCatalog


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


# Build the filter catalog using the RDKit filterCatalog module
def build_filter_catalog(filename):
    inhousefilter = pd.read_csv(filename)
    inhouseFiltersCat = FilterCatalog.FilterCatalog()

    for i in range(inhousefilter.shape[0]):
        mincount = 1

        if inhousefilter["MIN_COUNT"][i] != 0:
            mincount = int(inhousefilter["MIN_COUNT"][i])

        pname = inhousefilter["PATTERN_NAME"][i]
        sname = inhousefilter["SET_NAME"][i]
        pname_final = "{0}_min({1})__{2}__{3}__{4}".format(
            pname,
            mincount,
            inhousefilter["SEVERITY_SCORE"][i],
            inhousefilter["COVALENT"][i],
            inhousefilter["SPECIAL_MOL"][i],
        )
        fil = FilterCatalog.SmartsMatcher(
            pname_final, inhousefilter["SMARTS"][i], mincount
        )
        inhouseFiltersCat.AddEntry(FilterCatalog.FilterCatalogEntry(pname_final, fil))
        inhouseFiltersCat.GetEntry(i).SetProp("Scope", sname)

    return inhouseFiltersCat


if __name__ == "__main__":
    import sys
    import pickle

    catalog = build_filter_catalog(sys.argv[1])

    out_filename = sys.argv[2]

    with open(out_filename, "wb") as pfile:
        pickle.dump(catalog, pfile)

    with open(out_filename, "rb") as pfile:
        catalog = pickle.load(pfile)

    print(f"{type(catalog)=}")
