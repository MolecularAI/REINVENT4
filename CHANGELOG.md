# Changelog for REINVENT

This follows the guideline on [keep a changelog](https://keepachangelog.com/)


## [Unreleased]

### Changed

- Additional functionality for Mol2Mol sampling
- Fragment generators using transformers


## [4.0.17] 2023-11-17

### Fixed

- Bug in sampling related to the way the sampled.nlls object was treated. Now it is always pytorch tensor object without gradient on cpu.


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
