"""Extract poses by RMSD and CNN affinityo

Finds poses below Gnina's RMSD and keeps the one with the highest CNNaffinity
"""

import sys
from collections import defaultdict

from rdkit import Chem


filename = sys.argv[1]
mols = defaultdict(list)

suppl = Chem.SDMolSupplier(filename)

for mol in suppl:
    if mol:
        cnn_score = float(mol.GetProp("CNNscore"))

        if cnn_score > 0.9:
            score = float(mol.GetProp("CNNaffinity"))
            name = mol.GetProp("_Name")
            mols[name].append((score, mol))


best_mols = []

for mol in mols.values():
    s = sorted(mol, key=lambda e: e[0], reverse=True)
    best_mols.append(mol[0][1])

with Chem.SDWriter(sys.argv[2]) as sd_writer:
    for mol in best_mols:
        sd_writer.write(mol)
