import base64
import io
from io import BytesIO
from operator import itemgetter

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw as rkcd
from torch.utils.tensorboard import summary as tbs


def find_matching_pattern_in_smiles(list_of_mols: [], smarts_pattern=None) -> []:

    def orient_molecule_according_to_matching_pattern(molecule, pattern):
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                AllChem.Compute2DCoords(pattern_mol)
                AllChem.GenerateDepictionMatching2DStructure(molecule, pattern_mol, acceptFailure=True)
        except:
            pass

    matches = []
    if smarts_pattern is not None:
        for mol in list_of_mols:
            if mol is not None:
                match_pattern = mol.GetSubstructMatch(Chem.MolFromSmarts(smarts_pattern))
                orient_molecule_according_to_matching_pattern(mol, smarts_pattern) if len(match_pattern) > 0 else ()
                matches.append(match_pattern)
            else:
                no_pattern = ()
                matches.append(no_pattern)
    return matches


def padding_with_invalid_smiles(smiles, sample_size):
    diff = len(smiles) - sample_size
    if diff < 0:
        bulk = ["INVALID" for _ in range(-diff)]
        bulk_np = np.array(bulk)
        smiles = np.concatenate((smiles, bulk_np))
    return smiles


def check_for_invalid_mols_and_create_legend(smiles, score, sample_size):
    legends = []
    list_of_mols = []
    for i in range(sample_size):
        list_of_mols.extend([Chem.MolFromSmiles(smiles[i])])
        if list_of_mols[i] is not None:
            legends.extend([f"{score[i].item():.3f}"])
        elif list_of_mols[i] is None:
            legends.extend([f"This Molecule Is Invalid"])
    return list_of_mols, legends


def sort_smiles_by_score(score, smiles: []):
    paired = []
    for indx, _ in enumerate(score):
        paired.append((score[indx], smiles[indx]))
    result = sorted(paired, key=itemgetter(0), reverse=True)
    sorted_score = []
    sorted_smiles = []
    for r in result:
        sorted_score.append(r[0])
        sorted_smiles.append(r[1])
    return sorted_score, sorted_smiles


def mol_to_png_string(mol_list: [], molsPerRow=4, subImgSize=(300, 300), legend=None, matches=None):
    image = rkcd.MolsToGridImage(mols=mol_list, molsPerRow=molsPerRow, subImgSize=subImgSize, useSVG=False,
                                 legends=legend, highlightAtomLists=matches)
    buffered = BytesIO()
    image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    buffered.close()
    return str(img_str)[2:][:-1]  # trim on both ends b' and '


def mol_to_svg_string(mol_list: [], molsPerRow=4, subImgSize=(300, 300), legend=None, matches=None):
    image = rkcd.MolsToGridImage(mols=mol_list, molsPerRow=molsPerRow, subImgSize=subImgSize, useSVG=True,
                                 legends=legend, highlightAtomLists=matches)
    return image


def add_mols(writer, tag, mols, mols_per_row=1, legends=None, global_step=None, walltime=None, size_per_mol=(300, 300), pattern=None):
    """
    Adds molecules in a grid.
    """
    image = rkcd.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=size_per_mol, legends=legends, highlightAtomLists=pattern)
    add_image(writer, tag, image, global_step, walltime)


def add_image(writer, tag, image, global_step=None, walltime=None):
    """
    Adds an image from a PIL image.
    """
    channel = len(image.getbands())
    width, height = image.size

    output = io.BytesIO()
    image.save(output, format='png')
    image_string = output.getvalue()
    output.close()

    summary_image = tbs.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
    summary = tbs.Summary(value=[tbs.Summary.Value(tag=tag, image=summary_image)])
    writer.file_writer.add_summary(summary, global_step, walltime)


def fraction_valid_smiles(smiles):
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    fraction = 100 * i / len(smiles)
    return fraction