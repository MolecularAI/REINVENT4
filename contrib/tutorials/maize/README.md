# REINVENT4 tutorial with Maize

This tutorial provides an overview on how to use the [Maize](https://github.com/MolecularAI/maize) workflow manager with REINVENT4.  The examples make use of docking with [AutoDock GPU](https://github.com/ccsb-scripps/AutoDock-GPU).  The turorial is inspired by the results from the [disco crossdock benchmark](http://disco.csb.pitt.edu/Targets_top5.php).  Phospholipase A2 was chosen as a small and easy docking target using PDB IDs 1DB5 (isoform IIa) and 6G5J (isoform X).

## Preliminaries

- Install [maize-contrib](https://github.com/MolecularAI/maize-contrib) which also installs the Maize workflow manager.
- Install [AutoDock GPU](https://github.com/ccsb-scripps/AutoDock-GPU).
- Install [Gypsum-DL](https://github.com/durrantlab/gypsum_dl).


## Tutorials

The Maize workflow manager needs a global configuration file which maps executables and scripts, allows setting up environent variables and slurm (neither used in this tutorial), etc.  The actural workflow is provided in a YAML file.  Visualisation of the workflow can be done through the provided notebook (convert from Python to notebook with jypytext) and the ready PDF file.  The workflow is largely controlled through setting parameters in the REINVENT TOML file.

All tutorial runs use an intricate scoring setup which combines rule-of-5/Veber rules and group counts with docking.  Protein structures have been prepared with Schrodinger Maestro (the [HiQBind Dataset](https://figshare.com/articles/dataset/BioLiP2-Opt_Dataset/27430305) may be a viable alternative).  A script `prep.sh` outlines how to convert the input PDB to the required formats for AutoDock GPU.

### AutoDock GPU workflow

REINVENT passes the currently sampled SMILES to the workflow which undergo ligand preparation with Gypsum-DL.  In case Gypsum-DL fails (in about 5-10% of cases), RDKit takes over and adds hydrogens by simple valence checks.  The 3D structures are then passed to AutoDock GPU which returns the docking score to REINVENT for each of the original SMILES.  A separate branch of the workflow extracts and stores the best pose from docking.

## Classical Reinvent prior

Three runs with results (CSV, poses) are provided using the _de novo_ Reinvent prior.

### 1DB5

Runs REINVENT docking to 1DB5 to demonstrate how to use Maize with REINVENT.

### 6G5J

Runs REINVENT docking to 6G5J.

### 6G5J vs 1DB5

Demonstration on how off-target docking could work.  The run is ultimately able to find molecules binding more preferentially to isoform X as to isoform IIa as per docking score.  This shows that the setup works, at least at the technical level.  Structure would still need to be check for sanity and in connection with the non-docking scores.


## Libinvent

### 6G5J vs 8ZYP

Another demonstration of off-target docking, in this case with hERG (PDB ID 8ZYP).  The structures are contraint by the indole-2-carboxamide scaffold from 6G5J.  The run is quite well able to find differential ligands but, again, check results for sanity.
