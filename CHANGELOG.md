# Changelog for REINVENT

This follows the guideline on [keep a changelog](https://keepachangelog.com/)


## [Unreleased]

### Changed
- Fragment generators using transformers
- CAZP scoring component


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
