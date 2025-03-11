# TOML parameters

This is a summary of TOML parameters for each run mode.

## Sampling

Sample a number of SMILES with associated NLLs.


| Parameter          | Description                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------|
| run\_type          | set to "sampling"                                                                                    |
| device             | set the torch device e.g "cuda:0" or "cpu"                                                           |
| use\_cuda          | (deprecated) "true" to use GPU, "false" to use CPU                                                   |
| seed               | sets the random seeds for reproducibility               |
| json\_out\_config    | filename of the TOML file in JSON format                                                             |
| [parameters]       | starts the parameter section                                                                         |
| model\_file        | filename to model file from which to sample                                                          |
| smiles\_file       | filename for input SMILES for Lib/LinkInvent and Mol2Mol                                            |
| sample\_strategy   | Transformer models: "beamsearch" or "multinomial"                                                    |
| output\_file       | filename for the CSV file with samples SMILES and NLLs                                               |
| num\_smiles        | number of SMILES to sample, note: this is multiplied by the number of input SMILES                   |
| unique\_molecules  | if "true" only return unique canonicalized SMILES                                                    |
| randomize\_smiles  | if "true" shuffle atoms in input SMILES randomly                                                     |
| tb\_logdir         | if not empty string name of the TensorBoard logging directory                                        |
| temperature        | Mol2Mol only: default 1.0                                                                            |
| target\_smiles\_path | Mol2Mol only: if not empty, filename to provided SMILES, check NLL of generating the provided SMILES |


## Scoring

Interface to the scoring component.  Does not use any models.

| Parameter           | Description                                                                                             |
|---------------------|---------------------------------------------------------------------------------------------------------|
| run\_type           | set to "scoring"                                                                                        |
| device              | set the torch device e.g "cuda:0" or "cpu"                                                              |
| use\_cuda           | (deprecated) "true" to use GPU, "false" to use CPU                                                      |
| seed               | sets the random seeds for reproducibility (no effect in scoring mode)               |
| json\_out\_config   | filename of the TOML file in JSON format                                                                |
| [parameters]        | starts the parameter section                                                                            |
| smiles\_file        | SMILES filename, SMILES are expected in the first column                                                |
| output\_csv         | Name of the output CSV file to write                                                                    |
| [scoring\_function] | starts the section for scoring function setup                                                           |
| [[components]]      | start the section for a component within [scoring\_function] , note the double brackets to start a list |
| type                | "custom\_sum" for weighted arithmetic mean or "custom\_produc" for weighted geometric mean              |
| component\_type     | name of the component, FIXME: list all                                                                  |
| name                | a user chosen name for ouput in CSV files, etc.                                                         |
| weight              | the weight for this component                                                                           |


## Transfer Learning

Run transfer learning on a set of input SMILES.

| Parameter              | Description                                                   |
|------------------------|---------------------------------------------------------------|
| run\_type              | set to "transfer\_learning"                                    |
| device             | set the torch device e.g "cuda:0" or "cpu"                                                             |
| use\_cuda          | (deprecated) "true" to use GPU, "false" to use CPU                                                     |
| seed               | sets the random seeds for reproducibility               |
| json\_out\_config        | filename of the TOML file in JSON format                      |
| tb\_logdir             | if not empty string name of the TensorBoard logging directory |
| number\_of\_cpus       | optional parameter to control number of cpus for pair  generation. If not provided, only one CPU will be used. |
| [parameters]           | starts the parameter section                                  |
| num\_epochs            | number of epochs to run                                       |
| save\_every\_n\_epochs | save checkpoint file every N epochs                           |
| batch\_size            | batch size, note: affects SGD                                 |
| sample\_batch\_size    | batch size to calculate the sample loss and other statistics  |
| num\_refs              | number of references for similarity if > 0, DO NOT use with large dataset (> 200 molecules) |
| input\_model\_file     | filename of input prior model                                 |
| validation\_smiles\_file | SMILES file for validation                                  |
| smiles\_file           | SMILES file for Lib/Linkinvent and Molformer                  |
| output\_model\_file     | filename of the final model                                   |
| pairs.upper\_threshold | Molformer: upper similarity                                   |
| pairs.lower\_threshold | Molformer: lower similarity                                   |
| pairs.min\_cardinality | Molformer:                                                    |
| pairs.max\_cardinality | Molformer:                                                    |


## Staged Learning

Run reinforcement learning (RL) and/or curriculum learning (CL).  CL is simply a multi-stage RL learning.

| Parameter            | Description                                                                                                                    |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------|
| run\_type            | set to "staged\_learning"                                                                                                    |
| device               | set the torch device e.g "cuda:0" or "cpu"                                                                                     |
| use\_cuda            | (deprecated) "true" to use GPU, "false" to use CPU                                                                             |
| seed                 | sets the random seeds for reproducibility               |
| json\_out\_config    | filename of the TOML file in JSON format                                                                                       |
| tb\_logdir           | if not empty string name of the TensorBoard logging directory                                                                  |
| [parameters]         | starts the parameter section                                                                                                   |
| summary\_csv\_prefix | prefix for output CSV filename                                                                                                 |
| use\_checkpoint      | if "true" use diversity filter from agent\_file if present                                                                     |
| purge\_memories      | if "true" purge all diversity filter memories (scaffold, SMILES) after each stage                                              |
| prior\_file          | filename of the prior model file, serves as reference                                                                          |
| agent\_file          | filename of the agent model file, used for training, replace with checkpoint file from previous stage when needed              |
| batch\_size          | batch size, note: affects SGD                                                                                                  |
| unique\_sequences    | if "true" only return unique raw sequence (sampling)                                                                           |
| randomize\_smiles    | if "true" shuffle atoms in input SMILES randomly (sampling)                                                                    |
| [learning\_strategy] | start section for RL learning strategy                                                                                         |
| type                 | use "dap"                                                                                                                      |
| sigma                | sigma in the reward function                                                                                                   |
| rate                 | learning rate for the torch optimizer                                                                                          |
| [diversity\_filter]  | starts the section for the diversity filter, overwrites all stage DFs                                                          |
| type                 | name of the filter type: "IdenticalMurckoScaffold", "IdenticalTopologicalScaffold", "ScaffoldSimilarity", "PenalizeSameSmiles" |
| bucket\_size         | number of scaffolds to store before molecule is scored zero                                                                    |
| minscore             | minimum score                                                                                                                  |
| minsimilarity        | minimum similarity in "ScaffoldSimilarity"                                                                                     |
| penalty\_multiplier  | penalty penalty for each molecule in "PenalizeSameSmiles"                                                                      |
| [inception]          | starts the inception section                                                                                                   |
| smiles\_file         | filename for the "good" SMILES                                                                                                 |
| memory\_size         | number of SMILES to hold in inception memory                                                                                   |
| sample\_size         | number of SMILES randomly sampled from memory                                                                                  |
| [[stage]]            | starts a stage, note the double brackets                                                                                       |
| chkpt\_file          | filename of the checkpoint file, will be written on termination and Ctrl-C                                                     |
| termination          | use "simple", termination criterion                                                                                            |
| max\_score           | maximum score when to terminate                                                                                                |
| min\_steps           | minimum number of RL steps to avoid early termination                                                                          |
| max\_steps           | maximum number of RL steps to run, if maximum is hit **all** stages will be terminated                                         |
| [diversity\_filter]  | a per stage DF filter can be defined for each stage, global DF will overwrite this                                             |

The scoring functions are added as in scoring but prefixed with stage.
