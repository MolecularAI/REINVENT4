# Changelog for REINVENT

This follows the guideline on [keep a changelog](https://keepachangelog.com/)


## [Unreleased]

### Changed

- Additional functionality for Mol2Mol sampling
- Fragment generators using transformers


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
