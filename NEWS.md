New in REINVENT 4.1
===================

For details, see CHANGELOG.md.

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
