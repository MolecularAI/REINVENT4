CAZP support
============


Description
-----------

The files in this directory are the support files for the CAZP scoring
component in REINVENT.  CAZP is an interface to
(AiZynthFinder)[https://github.com/MolecularAI/aizynthfinder], a tool for
retrosynthesis planning (see
(documentation)[https://molecularai.github.io/aizynthfinder/]).


Requirements
------------

Installed (AiZynthFinder)[https://github.com/MolecularAI/aizynthfinder].
AiZynthFinder preferably runs on multiple CPUs/cores.


CAZP configuration
------------------

The main CAZP configuration is a JSON file pointed to by the environment
variable `CAZP_PROFILES`.  This file must contain two fields.
  1. `base_aizynthfinder_config`: the AiZynthFinder configuration in YAML format
  2. `custom_aizynth_command`: a script to run AiZynthFinder

Additional fields can be added to extend the AiZynthFinder configuration.
