# Changelog for REINVENT

This follows the guideline on [keep a changelog](https://keepachangelog.com/)


## [Unreleased]

### Changed

- CAZP scoring component


## [4.5.11] 2024-11-18

### Changed

- Convert float nan and infs to valid json format before remote reporting


## [4.5.10] 2024-11-16

### Added

- optional tautomer canonicalisation in data pipeline


## [4.5.9] 2024-11-07

### Fixed

- read configuration file from stdin


## [4.5.8] 2024-11-07

### Changed

- refactor of top level code


## [4.5.7] 2024-11-07

### Fixed

- check if DF is set


## [4.5.6] 2024-11-05

### Added

- YAML configuration file reader


## [4.5.5] 2024-11-05

### Added

- Logging of configuration file absolute path

### Changed

- Automatic configuration file format detection


## [4.5.4] 2024-10-28

### Added

- Exponential decay transform

### Fixed

- Ambiguity in parsing optional parameters with multiple endpoints and multiple optional parameters


## [4.5.3] 2024-10-23

### Added

- component-level parameters for scoring components


## [4.5.2] 2024-10-23

### Added

- executable module: can run `python -m reinvent`


## [4.5.1] 2024-10-23

### Added

- SIGUSR1 for controlled termination


## [4.5.0] 2024-10-08

### Added

- PepInvent in Sampling and Staged learning mode with example toml config provided 
- PepInvent prior


## [4.4.37] 2024-10-07

### Fixed

- Atom map number removal for Libinvent sampling dropped SMILES


## [4.4.36] 2024-09-27

### Added

- Stage number for JSON to remote monitor

### Changed

- Relaxed dependencies


## [4.4.35] 2024-09-26

### Added

- Terminate staged learning on SIGTERM and check if running in multiprocessing environment

### Changed

- ValueError for all scoring components such that the staged learning handler can handle failing components


## [4.4.34] 2024-09-16

### Fixed

- SMILES in DF memory were wrongly computed


## [4.4.33] 2024-09-14

### Fixed

- run-qsartuna.py: convert ndarray to list to make it JSON serializble


## [4.4.32] 2024-09-13

### Fixed

- PMI component: check for embedding failure in RDKit's conformer generator


## [4.4.31] 2024-09-13

### Fixed

- Dockstream component wrongly quoted the SMILES string
- Diversity filter setup in config was ignored


## [4.4.30] 2024-09-12

### Fixed

- Fixed config reading bug for DF


## [4.4.29] 2024-09-05

### Changed

- Changed Molformer sampling valid and unique from percentage to fraction on tensorboard


## [4.4.28] 2024-08-29

### Fixed

- Fixed incorrect tanimoto similarity log in Mol2Mol sampling mode


## [4.4.27] 2024-07-23

### Fixed

- Corrected typo in Libinvent report


## [4.4.26] 2024-07-21

### Fixed

- Report for sampling returned np.array which is incompatibile with JSON serialization


## [4.4.25] 2024-07-19

### Fixed

- Allowed responder as an optional input in scoring input validation


## [4.4.24] 2024-07-19

### Fixed

- Fixed remote for Libinvent
- Batchsize defaults to 1 for TL


## [4.4.23] 2024-07-18

### Fixed

- Added temperature parameter in Sampling and RL config validation 


## [4.4.22] 2024-07-10

### Fixed

- Scalar return value from make\_grid\_image()


## [4.4.21] 2024-07-10

### Changed

- Removed labels from Libinvent generated molecules before passing to scoring components: maize, icolos, dockstream and ROCS
- Write out clean SMILES without labels to CSV and tensorboard in staged learning and sampling run modes
- Simplified update code for Libinvent RL


## [4.4.20] 2024-07-10

### Fixed

- Added missing layer\_normalization when getting Reinvent RNN model parameters


## [4.4.19] 2024-07-08

### Fixed

- Log likelihood calculation is handled more efficiently


## [4.4.18] 2024-07-05

### Fixed

- Added missing tag to legacy TanimotoDistance component


## [4.4.17] 2024-07-05

### Added

- Custom RDKit normalization transforms hard-coded or from file


## [4.4.16] 2024-07-01

### Fixed

- Safe-guard against invalid SMILES in SMILES file/CSV reading


## [4.4.15] 2024-06-27

### Added

- Allow dot SMILES fragment separator for Lib/Linkinvent input


## [4.4.14] 2024-06-27

### Changed

- Renamed pipeline options keep\_isotopes to keep\_isotope\_molecules

### Added

- New data pipeline options: uncharge, kekulize, randomize\_smiles


## [4.4.13] 2024-06-27

### Fixed

- Parallel logging in data pipeline preprocessor (partial only)


## [4.4.12] 2024-06-25

### Added

- Libinvent based on unified Transformer

### Fixed

- Error handling for unsupported tokens in RNN-based Libinvent


## [4.4.11] 2024-06-17

### Added

- Parallel processing of regex and rdkit filters

### Chamged

- RDKit filter uses simpler functions not compound functions


## [4.4.10] 2024-06-05

### Changed

- RL max\_score is optional and has a default of 1.0


## [4.4.9] 2024-06-05

### Added

- Parallel implementation of chemistry filter in data pipeline


## [4.4.8] 2024-06-05

### Fixed

- Added multinomial as default sampling strategy for Transformer. 


## [4.4.7] 2024-06-04

### Changed

- Chemistry filter: customizable normalization, do not use RDKit logger


## [4.4.6] 2024-06-04

### Changed

- Reworked remote responder


## [4.4.5] 2024-06-03

### Changed

- Reporting for sampling runmode


## [4.4.4] 2024-06-03

### Fixed

- Minor fix to DF validation


## [4.4.3] 2024-06-03

### Added

- Chemistry filter for data pipeline


## [4.4.2] 2024-05-30

### Fixed

- Check if agent is in RL state info to avoid unnecessary exception


## [4.4.1] 2024-05-30

### Changed

- Toplevel validation of config file to detect extra sections


## [4.4.0] 2024-05-30

### Added

- Initial support for data pipeline


## [4.3.26] 2024-05-30

### Added

- Example scoring component script for RAScore
- Reimplentation of Conversion class for backward compatibility


## [4.3.25] 2024-05-29

### Fixed

- Various fixes in TLRL notebook


## [4.3.24] 2024-05-27

### Changed

- Scoring runmode now also allows import of scoring components


## [4.3.23] 2024-05-25

### Changed

- More consistent JSON config writing (includes imported scoring functions)
- Better handling of value mapping


## [4.3.22] 2024-05-24

### Fixed

- All scoring components need to compute the number of endpoints, added where sensible


## [4.3.21] 2024-05-24

### Fixed

- Filter out invalid fragments for Lib/Linkinvent


## [4.3.20] 2024-05-22

### Changed

- Lifted static methods to module level functions

### Removed

- More chemistry code


## [4.3.19] 2024-05-21

### Removed

- Unused chemistry code and associated tests


## [4.3.18] 2024-05-17

### Added

- Chained reporters for RL

### Fixed

- Compatibility support for model file metadata: dict vs dataclass


## [4.3.17] 2024-05-15

### Changed

- Various cosmetic fixes to TB output

### Fixed

- TL responder: validation loss was reported as sampled loss
- Add metadata when "metadata" files is empty


## [4.3.16] 2024-05-14

### Changed

- Code clean-up in Reinvent model and RNN code
- Global pydantic configuration

### Fixed

- Affected test cases after code rearrangement


## [4.3.15] 2024-05-13

### Added

Various code improvements.
- Metadata writing for all created RL and TL models
- Chained reporters
- Prior registry
- Config validation


## [4.3.14] 2024-05-06

### Added

- Write additional information to RL CSV
  - For Mol2Mol, add _Input\_SMILEs_
  - For Linkinvent, add _Warheads_ and _Linker_
  - For Libinvnet, add _Input\_Scaffold_ and _R-groups_


## [4.3.13] 2024-05-06

### Added

- Notebook: plot of reverse sigmoid transform


## [4.3.12] 2024-05-06

### Added

- Stages can now define their own diversity filters.  Global filter always overwrites stage settings.  Currently no mechanism to carry over DF from previous stage, use single stage runs.


## [4.3.11] 2024-04-30

### Changed

- TanimotoSimilarity replaces TanimotoDistance

### Deprecated

- TanimotoDistance: computes actually a similarity and TanimotoSimilarity


## [4.3.10] 2024-04-29

### Changed

- ChemProp scoring component now supports multitask models


## [4.3.9] 2024-04-28

### Added

- Optional [scheduler] section for TL


## [4.3.8] 2024-04-24

### Fixed

- LibInvent: fixed issue with multiple R-groups on one atom
- ReactionFilter: selective filter will now function correctly as filter


## [4.3.7] 2024-04-22

### Added

- Notebook: a more complete RL/TL demo


## [4.3.6] 2024-04-22

### Fixed

- Fixed crash when all molecules in batch are filtered out


## [4.3.5] 2024-04-18

### Changed

- Code clean-up in create\_adapter()

### Fixed

- Import mol2mol vocabulary rather than copying the file


## [4.3.4] 2024-04-16

### Added

- Write invalid SMILES unchanged to RL CSV


## [4.3.3] 2024-04-16

### Added

- Notebook: demo on how to analyse RL CSV


## [4.3.2] 2024-04-15

### Added

- Dataclass validation for scoring component parameters

### Fixed

- Datatype in MatchingSubstructure's Parameters: only single SMARTS is allowed


## [4.3.1] 2024-04-15

### Added

- Notebook to demo simple RL run, TensorBoard visualisation and TensorBoard data extraction.


## [4.3.0] 2024-04-15

### Added

- Linkinvent based on unified Transformer model supported by RL and sampling.  Both beam search and multinomial sampling are implemented.


## [4.2.13] 2024-04-12

### Fixed

- downgraded Chemprop to 1.5.2 and sklearn to 1.2.2 to retain backward compatibility


## [4.2.12] 2024-04-10

### Changed

- New default torch device setup from PyTorch 2.x

### Added

- Config parameter "device" to explicitly set torch device e.g. "cuda:0"


## [4.2.11] 2024-04-08

### Fixed

- Fixed unknown token handling for Mol2mol TL


## [4.2.10] 2024-04-05

### Fixed

- Fixed dataloader for TL to use incomplete batch 


## [4.2.9] 2024-04-04

### Fixed

- Skip hash check in metadata if no metadata in model file


## [4.2.8] 2024-04-03

### Added

- Mol2Mol supports unknown tokens for all the priors



## [4.2.7] 2024-03-27

### Added

- Optional randomization in all TL epochs for Reinvent


## [4.2.6] 2024-03-21

### Fixed

- Return from make\_grid\_image()


## [4.2.5] 2024-03-20

### Added

- Log network parameters


## [4.2.4] 2024-03-20

### Changed

- Reworked TL code: clean-up, image layout, graph for Reinvent

### Added

- Added options and better statistics for TL: valid SMILES, duplicates
- Disableable standardization

### Removed

- KL divergence in TB output

### Fixed

- Batch size calculation


## [4.2.3] 2024-03-14

### Fixed

- Vocabulary for mol2mol and reinvent is saved as dictionary


## [4.2.2] 2024-03-13

### Fixed

- Sigmoid functions in scoring have now a stable implementation


## [4.2.1] 2024-03-12

### Removed

- Long chain (SMARTS for 5 aliphatic carbons) check


## [4.2.0] 2024-03-08

### Added

- Unified transformer code to faciliate new model designs


## [4.1.16] 2024-03-07

### Fixed

- filters now apply to transformed scores 


## [4.1.15] 2024-03-07

### Fixed

- Minor change inception filter: cleaner way of handling internal SMILES store


## [4.1.14] 2024-03-06

### Added

- Updated script for creating an empty classical Reinvent model

### Fixed

- Memory bug in TL related to similarity calculation: made this optional


## [4.1.13] 2024-03-05

### Added

- Allowed runs with only filter/penalty components 


## [4.1.12] 2024-03-05

### Added

- Better logging for Reinvent standardizer


## [4.1.11] 2024-02-27

### Fixed

- Inception filters out SMILES containing tokens that are not compatible with the prior


## [4.1.10] 2024-02-26

### Fixed

- Numerically stable double sigmoid implementation
- Number of CPUs for TL (Mol2Mol) is now 1


## [4.1.9] 2024-02-22

### Fixed

- Save models with no metadata


## [4.1.8] 2024-02-20

### Fixed

- Tensorboard histogram bug fixed again


## [4.1.7] 2024-02-20

### Fixed

- TL is now running for the expected `num_epochs`


## [4.1.6] 2024-02-19

### Fixed

- Get model\_type from save\_dict prior model correctly


## [4.1.5] 2024-02-13

### Fixed

- Staged learning does not allocate GPU memory if device is set to CPU


## [4.1.4] 2024-02-06

### Added

- Prior model files have been tagged with meta data
- Model files read in are checked for integrity


## [4.1.3] 2024-02-06

### Fixed

- Tab reader unit tests now uses mocks for open
- Wite correctly CSV scoring file when from one columns SMILES file


## [4.1.2] 2024-02-04

### Fixed

- Scoring filter components work as filters again


## [4.1.1] 2024-02-02

### Added

- CSV and SMILES file reader for the scoring run mode, will retain all columns form the input and write to output CSV


## [4.1.0] 2024-01-26

### Added

- Tobias Ploetz' (Merck) REINFORCE implementations of the DAP, MAULI and MASCOF RL reward strategies


## [4.0.36] 2024-01-22

### Added

- Check if RDKit descriptor names are valid


## [4.0.35] 2024-01-19

### Fixed

- Filename issue on Windows which lead to termination


## [4.0.34] 2024-01-17

### Added

- General scoring component for all 210 RDKit descriptors


## [4.0.33] 2023-12-14

### Added

- Optional cwd for run\_command()


## [4.0.32] 2023-12-12

### Fixed

- Collect _all_ names for remote monitoring\*


## [4.0.31] 2023-12-08

### Fixed

- Pass data to request as dict rather than a JSON string


## [4.0.30] 2023-12-07

### Fixed

- Pair generator multiprocessing in TL is supported on Linux, Windows, and MacOS

### Changed

- The number of cpus is optional and could be specified in the toml/json configuration file through the parameter `number_of_cpus`


## [4.0.29] 2023-12-07

### Fixed

- Added missing second mask in inception call to scoring function
- Fixed cuda out of memory for Reinvent batch sampling


## [4.0.28] 2023-12-05

### Fixed

- Handle the case when there are no non-cached SMILES and thus the scpring function does not need to run.
- Improved type safety in `value_mapping`


## [4.0.27] 2023-12-01

### Added

- Number of cpus can be specified in toml/json config files for TL jobs


## [4.0.26] 2023-12-01

### Fixed

- Check for CUDA before checking GPU memory otherwise will fail on CPU
- Removed obsolete code which broke TL with Reinvent
- Windows support: correct signal handling


## [4.0.25] 2023-11-29

### Added

- Scoring component MolVolume to compure molecular volume via RDKit


## [4.0.24] 2023-11-28

### Changed

- Minimal SMILES pre-processing for scoring to allow keeping of stereochemistry and only choose largest fragment, and use the general RDKit cleanup/sanitation/hydrogen.  Skip heavy filtering on molecules size, allowed atoms, tokens, vocabulary, etc.  This faciliates situation where only scoring is desired.


## [4.0.23] 2023-11-28

### Changed

- Allow zero weights to only display a component score.  This will have no effect on aggregation but the component score is still computed. So, be careful with computationally expensive components.


## [4.0.22] 2023-11-28

### Changed

- Flag to purge diversity filter memories after each staged learning stage.  This is useful in multiple stage runs and is equivalent to `use_checkpoint` for single stage reruns.


## [4.0.21] 2023-11-23

### Changed

- The CSV file from RL has controlled output precision: 7 for total score and transformed scores, 4 for all other floating point values

### Fixed

- Critical: all scores of duplicate SMILES including the first occurence where set to zero rather than the computed value
- Scores of duplicates are now correctly copied over from first occurence


## [4.0.20] 2023-11-20

### Fixed

- All tests support both cpu and gpu


## [4.0.19] 2023-11-20

### Changed

- Contidional import of `resource` to allow running on Windows


## [4.0.18] 2023-11-17

### Added

- Some rudimentary information on GPU memory usage in staged learning


## [4.0.17] 2023-11-17

### Fixed

- Bug in sampling related to the way the sampled.nlls object was treated. Now it is always pytorch tensor object without gradient on cpu


## [4.0.16] 2023-11-15

### Fixed

- Moved TPSA to separate component to enable TPSA calculation for polar S and P, the original RDKit implementation does not consider those atoms and the default is still to leave those out from TPSA calculation


## [4.0.15] 2023-11-15

### Fixed

- Issue with NaNs being in the raw and transformed code that would not allow to compute the mean


## [4.0.14] 2023-11-13

### Fixed

- Explicit serialization of JSON string because the internal one from requests may fail


## [4.0.13] 2023-11-09

### Fixed

- Added a patch to fix a bug on the native implementation of Pytorch related to the histogram functionality of the Tensorboard report


## [4.0.12] 2023-11-09

### Changed

- Added a check which raises an exception if the user enters scaffolds with multiple attachement points connected to the same atom (Libinvent, Linkinvent). This
will be lifted in a future update

### Fixed

- Fixed report format (TL)


## [4.0.11] 2023-11-02

### Fixed

- Normalize SMILES before passing to string-based scoring component because the SMILES may still contain lables (Libinvent)


## [4.0.10] 2023-10-27

### Fixed

- Fixed fragment effective length ratio error when fragment has single atom (e.g. "[*]N[*]") with max graph length=0 


## [4.0.9] 2023-10-25

### Changed

- For Compound Sampling, removed explicit dependency on matplotlib: report histgram and scatter plot to tensorboard only if matplotlib is available. 


## [4.0.8] 2023-10-23

### Changed

- For Compound Sampling, introduced a new parameter unique\_molecules to do canonicalize smiles deduplication instead of using sequemce deduplication which can be confusing


## [4.0.7] 2023-10-20

### Deprecated

- warn if sequence deduplication is requested


## [4.0.6] 2023-10-18

### Fixed

- TL integration test fixed (no impact on GUI or core)


## [4.0.5] 2023-10-17

### Added

- TL reporting of epochs

### Changed

- multiple scoring components are reported as means again rather than as lists


## [4.0.3] 2023-10-16

### Fixed

- fixed graph length fragment component


## [4.0.2] 2023-10-06

### Fixed

- fixes for fragment scoring components


## [4.0.1] 2023-10-03

### Changed

- scores reported for filters and penalties as in REINVENT3


## [4.0.0] 2023-09-12

### Added

- initial release of REINVENT4

### Deprecated
- RL rewards MASCOF, MAULI, SDAP (inefficient in practice)
- sequence deduplication (interferes with diversity filter and SMILES deduplication)


## Older Releases

### [3.2], [3.1], [3.0], [2.1], [2.0]

[Reinvent on GitHub](https://github.com/MolecularAI/Reinvent)
