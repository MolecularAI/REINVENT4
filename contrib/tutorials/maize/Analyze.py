# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # REINVENT4 tutorial using Maize
#
#

# %%
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import mols2grid

from IPython.display import display

sns.set(style="darkgrid")
pd.set_option("display.precision", 3)


# %% [markdown]
# ## Utility functions
#
# Useful functions to display molecules in a grid and to show a 3D conformation of a molecule.

# %%
def show_mol_grid(df, cols=None, smiles_col='SMILES', mol_col=None, transform={}):

    if cols:
        subset = ['img'].append(cols)
    else:
        subset = ['img']

    if mol_col:
        mol_view = mols2grid.display(df, mol_col=mol_col,
                                     subset=subset, n_rows=5, n_cols=6, selection=False,
                                     transform=transform, useSVG=True, size=(150, 150))
    else:
        mol_view = mols2grid.display(df, smiles_col=smiles_col,
                                     subset=subset, n_rows=5, n_cols=6, selection=False,
                                     transform=transform, useSVG=True, size=(150, 150))

    display(mol_view)

import py3Dmol

def show_mol_3d(mol):
    view = py3Dmol.view(width=500, height=500)
    view.removeAllModels()
    
    IPythonConsole.addMolToView(mol, view, confId=0)
    
    view.zoomTo()
    view.show()
    
import nglview as ngl  # requires nodejs from conda
import ipywidgets as widgets

class ShowLigand:
    def __init__(self, view, mols):
        self.view = view
        self.mols = mols
        self.id = None
        
        ligand_dropdown = widgets.Dropdown(
            options=[(self.mols[i].GetProp("SMILES"), i) for i, m in enumerate(self.mols)],
            description="SMILES:",
            style={'description_width': 'initial'},
        )

        widgets.interactive_output(self.show_ligand, {"index": ligand_dropdown})

        display(widgets.VBox([ngl_view, ligand_dropdown]))

    def show_ligand(self, index):
        if self.id is not None:
            try:
                self.view.remove_component(self.id)
            except Exception:
                pass
    
        self.id = self.view.add_component(self.mols[index])
        self.view[1].add_ball_and_stick


# %% [markdown]
# ## Exploring the data
#
# The data has been post-process after the REINVENT to filter down the original set of 60,000 molecules.  The filter removes invalid molecules, duplicates and discards all molecules with a score lower than a threshold in any of its individual scores.  We also assume that a predicted free energy (the docking score really) of -14.0 kcal/should and smaller provides sufficiently reasonable docked poses for further processing.  The 3D conformations/poses of the final subset are stored in a separate SD file.

# %%
df = pd.read_csv("Reinvent/1DB5/top.csv")

# %% [markdown]
# ### Distribution of the AutoDock score
#
# Keep in mind that the original data has been filtered so naturally there will be an upper bound of the score at the originally chosen cutoff.

# %%
sns.histplot(data=df, x="AutoDockGPU (raw)", stat="density", label="AD Score")

plt.legend()
plt.show()

# %% [markdown]
# ### Distribution of calculated logP
#
# We look here only at the calculated logP which is often taken as a proxy for solubility and permeability.  The user is encouraged to explore the other scores as well.  See below for all the column names.

# %%
sns.histplot(data=df, x="logP (<=5) (raw)", stat="density", label="logP")

plt.legend()
plt.show()

# %% [markdown]
# ###  All columns from the CSV file

# %%
df.columns

# %% [markdown]
# ### 2D display of filtered molecules
#
# A 6-by-4 grid showing the 2D structure of each molecule together with all metadata from REINVENT, see the little "i" button in the top right of each box.

# %%
show_mol_grid(df, smiles_col='SMILES', cols=["AutoDockGPU (raw)"])

# %% [markdown]
# ## Exploring the Gnina poses
#
# The filtered poses from above have been subjected to local optimization with Gnina using the default CNN score for rescoring.  These poses are then further filtered by the `CNNscore` which basically is Gnina's confidence in how close it thinks the true pose is (within 2A).  This helps to further filter down the molecules.  There assumption here is made that Gnina provides better poses and scores for downstream tasks.

# %%
suppl = Chem.SDMolSupplier("Reinvent/1DB5/best_pose_local.sdf")
mols = []

for mol in suppl:
    if mol:
        adgpu_score = mol.GetProp("m_score__min__energy")
        gnina_score = mol.GetProp("minimizedAffinity")
        gnina_cnn_score = mol.GetProp("CNNscore")
        gnina_cnn_affinity = mol.GetProp("CNNaffinity")
        gnina_rmsd = mol.GetProp("minimizedRMSD")
        mols.append((adgpu_score, gnina_score, gnina_cnn_score, gnina_cnn_affinity, gnina_rmsd, mol))

best = pd.DataFrame(mols, columns=["AD GPU score", "Gnina score", "Gnina CNN score",
                                   "Gnina CNN affinity", "Gnina RMSD", "Mol"])

# %%
best[["AD GPU score", "Gnina score", "Gnina CNN score", "Gnina CNN affinity", "Gnina RMSD"]].round(3)

# %%
show_mol_grid(best, mol_col="Mol", cols=["all"])

# %% [markdown]
# ### Exploring 3D poses

# %%
sel_mol = best.Mol.iloc[3]
sel_mol

# %%
show_mol_3d(sel_mol)

# %% [markdown]
# #### Show a specific pose in the 3D complex with the receptor
#
# This uses the interactive [NGL viewer](https://nglviewer.org/nglview/latest/) embedded here in a notebook cell. You can switch to full-screem when pressing the little error in the top right of the rendering.

# %%
ngl_view = ngl.show_file("adgpu_prepare/1DB5_apo.pdb")
ngl_view.add_representation('cartoon', selection='protein')
ngl_view.add_representation('ball+stick', selection='CA')
#ngl_view.add_representation('surface', selection='protein')

sl = ShowLigand(ngl_view, best.Mol)

# %%
