"""Notebook support code"""

import glob
from io import BytesIO

from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import mols2grid

sns.set(style="darkgrid")


def load_tb_data(wd, prefix="tb_"):
    """Load TensorBoard data

    The data is assume to be in the first file starting with 'events'.
    Only loads scalars (all) and the last image.

    :param wd: directory containing a TB directory
    :param prefix: prefix of the TB directory
    :returns: TB event accumulator
    """

    events_filename = None

    for filename in glob.glob(f"{wd}/{prefix}*/*"):
        if "events" in filename:
            events_filename = filename
            break

    if not events_filename:
        return

    ea = event_accumulator.EventAccumulator(
        events_filename,
        size_guidance={
            event_accumulator.IMAGES: 1,  # only final image
            event_accumulator.SCALARS: 0,  # scalars for all steps
        },
    )

    ea.Reload()

    return ea


def plot_scalars(ea):
    """Plot all TB scalars

    :param ea: TB event accumulator
    :returns: TB data as dataframe
    """

    labels = [label for label in ea.Tags()["scalars"] if not "(raw)" in label]
    nlabels = len(labels)

    fig, ax_top = plt.subplots(nrows=nlabels, ncols=1, figsize=(5, 15))
    fig.tight_layout()

    dfs = []

    for n, label in enumerate(labels):
        df = pd.DataFrame(ea.Scalars(label))
        df = pd.DataFrame(df.rename(columns={"value": label})[label])
        dfs.append(df)

        ax = sns.scatterplot(data=df, x=list(df.index.values), y=label, ax=ax_top[n])
        ax.set(xlabel=label, ylabel="Score")

    plt.show()

    return pd.concat(dfs, join="inner", axis=1)


def get_image(ea):
    """Display the TB image

    Assumes only a single image group and image is available.

    :param ea: TB event accumulator
    :returns: PIL image
    """

    image_label = ea.Tags()["images"][0]
    image_data = ea.Images(image_label)[-1]
    bio = BytesIO(image_data.encoded_image_string)
    img = Image.open(bio)

    return img


def create_mol_grid(df, cols=[], smiles_col="SMILES", transform={}):
    """Create a mol2grid grid

    :param df: Pandas DataFrame with SMILES and other data
    :param cols: columns to show
    :param smiles_col: name of the SMILES column in the DataFrame
    :param transform: transform for the column strings
    :returns: a mol view
    """

    subset = ["img"].append(cols)

    mol_grid = mols2grid.MolGrid(df, smiles_col=smiles_col, useSVG=True, size=(150, 150))
    mol_view = mol_grid.display(
        subset=subset, n_rows=5, n_cols=6, selection=False, transform=transform
    )

    return mol_view
