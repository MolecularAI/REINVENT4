New in REINVENT 4.6
===================

* All priors now on [Zenodo](https://doi.org/10.5281/zenodo.15641296)
* CAZP scoring component: interface to AiZynthFinder
* Optional SMARTS filter for sampling to remove unwanted patterns e.g. elements
* Output SMILES state for sampling
* iSIM similarty support for RL in TensorBoard
* Parallel scoring
* Ring system precedence based scoring component (from Pat Walters)
* Multiple endpoints for ExternalProcess scoring component
* ChemProp is still 1.x: rdkit\_2d\_normalized is obsolete (but needed and ignored) use features="rdkit\_2d\_normalized"
* Write out metadata to CSV e.g. custom alert matching pattern
* GPU memory reporting for TL
* Terminate if invalid tokens are found in the input
* Additions and fixes to PepInvent e.g CamSol scoring component
* Updates to the data pipeline e.g. optional InchiKey deduplication
* Various code improvements and fixes


New in REINVENT 4.5
===================

For details see CHANGELOG.md.

* PepInvent: transformer (SMILES) based peptide generator and prior model
* Temperature factor parameter (transformer generators) for sampling and RL
* Support script run-qsartuna.py to play QSARtuna models in external environment
* Component-level parameters for scoring components
* Renamed Qptuna scoring component to [QSARtuna](https://github.com/MolecularAI/QSARtuna)
* Staged learning terminates on SIGTERM (Ctrl-C) and writes out checkpoint file
* SIGUSR1 for graceful termination of staged learning runs
* Relaxed dependencies to accomodate install of other software in same environment e.g. QSARtuna
* Updated some dependencies e.g. PyTorch (now at version 2.4.1)
* New notebook in contrib demoing docking with DockStream and OpenEye
* YAML configuration file reader
* Configuration file format is automatically detected from filename extension
* Various code improvements and fixes


New in REINVENT 4.4
===================

For details see CHANGELOG.md.

* Transformer based Libinvent
* Prior registry to load internal priors more easily
* Strict validation of input configuration to ensure consistency
* Better JSON configuration file writing
* Metadata writing for all created RL and TL models
* Import functionality for scoring runmode
* Stages in staged learning can have their own diversity filters
* More memory efficient transformer models to handle larger numbers of input SMILES
* Additional (fragment) SMILES written to staged learning CSV
* TanimotoDistance renamed to TanimotoSimilarity
* Support for ChemProp multitask models: requires param.target\_column
* Allow dot SMILES fragment separator for Lib/Linkinvent input
* Optional [scheduler] section for TL
* Example support script for RAScore
* A more complete RL/TL demo notebook
* Experimental data pipeline to preprocess SMILES for prior creation
* Various code improvements and fixes


New in REINVENT 4.3
===================

For details see CHANGELOG.md.

* Upgrade to **PyTorch 2.2**: rerun `pip install -r requirements-linux-64.lock`
* 2 new **notebooks** demoing Reinvent with reinforcement learning and also transfer learning, includes TensorBoard visualisation and basic analysis
* New Linkinvent model code based on unified transformer
* New PubChem Mol2Mol prior
* Unknown token support for PubChem based transformer models
* New "device" config parameter to allow for explicit device e.g. "cuda:0"
* Optional SMILES randomization in every TL epoch for Reinvent
* Dataclass parameter validation for most scoring components
* Invalid SMILES are now written to the reinforcement learning CSV
* Code improvements and fixes


New in REINVENT 4.2
===================

For details see CHANGELOG.md.

* Reworked TL code with added options and statistics
* Standardization can be switched off in TL (useful in new prior creation)
* Similarity calculation in TL made optional
* Updated script for empty classical Reinvent model creation
* Allow runs with only filter/penalty components
* Stable sigmoid functions
* Removed long chain check in SMILES processing
* Unified transformer code
* Filter apply to transformed scores
* Better memory handling in inception
* Better logging for Reinvent standardizer
* Inception filters for tokens not compatible with the prior
* Number of CPUs for TL (Mol2Mol pair generation) is 1 by default
* Tensorboard histogram bug fixed again
* Code improvements and fixes


New in REINVENT 4.1
===================

For details see CHANGELOG.md.

* Scoring component MolVolume
* Scoring component for all 210 RDKit descriptors
* CSV and SMILES file reader for the scoring run mode
* Tobias Ploetz' (Merck) REINFORCE implementations of the DAP, MAULI and MASCOF RL reward strategies
* Number of CPUs can be specified for TL jobs: useful for Windows
* All prior models tagged with metadata and checked for integrity
* Code improvements and fixes


New in REINVENT 4.0
===================

* Combined RL/CL (staged learning)
* New transformer model for molecule optimization
* Full integration of all generators with all algorithmic frameworks (RL, TL)
* Reworked scoring component utilizing a plugin mechanism for easy extension
* TOML configuration file format in addition to JSON: note that format is incompatible with release 3.x
* Major code rewrite
* Single repository for all code
