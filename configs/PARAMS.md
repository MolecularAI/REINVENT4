# TOML parameters

Summary of TOML parameters for each run mode.  All run modes share a small set
of top-level keys, followed by mode-specific sections.


## Top-level keys (all run modes)

| Parameter         | Description                                                         |
|-------------------|---------------------------------------------------------------------|
| `run_type`        | One of `"sampling"`, `"scoring"`, `"transfer_learning"`, `"staged_learning"`, `"enumeration"`. |
| `device`          | PyTorch device string, e.g. `"cuda:0"` or `"cpu"`.                 |
| `json_out_config` | If set, write the parsed TOML config out to this JSON file.         |
| `tb_logdir`       | TensorBoard log directory name (empty string disables logging).     |
| `use_cuda`        | *Deprecated.* Use `device` instead.                                 |


---

## Sampling

Sample SMILES (and their NLLs) from a model.  The generator type is determined
by the model file; a SMILES seed file is required for LibInvent, LinkInvent,
Mol2Mol and Pepinvent.

**Section**: `[parameters]`

| Parameter            | Default          | Description                                                                                        |
|----------------------|------------------|----------------------------------------------------------------------------------------------------|
| `model_file`         | required         | Path to the model file to sample from.                                                             |
| `smiles_file`        | —                | Seed SMILES file. Required for LibInvent, LinkInvent, Mol2Mol and Pepinvent (see notes below).     |
| `num_smiles`         | required         | Number of SMILES to sample. For seed-based models this is the count *per input SMILES*.            |
| `output_file`        | `"samples.csv"`  | Output CSV file with sampled SMILES and their NLL values.                                          |
| `unique_molecules`   | `true`           | If `true`, deduplicate and canonicalize output SMILES.                                             |
| `randomize_smiles`   | `true`           | If `true`, randomize atom order in input SMILES before sampling.                                   |
| `isomeric_smiles`    | `false`          | If `true`, generate isomeric SMILES (transformer models always produce isomeric SMILES).           |
| `sample_strategy`    | `"multinomial"`  | Transformer models only: `"multinomial"` or `"beamsearch"` (deterministic).                       |
| `temperature`        | `1.0`            | Sampling temperature for multinomial strategy.                                                     |
| `target_smiles_path` | `""`             | If non-empty, compute the NLL of generating the provided SMILES instead of sampling freely.        |

**Optional filter section** `[filter]`: restrict sampled SMILES with a SMARTS blocklist.

| Parameter | Description                              |
|-----------|------------------------------------------|
| `smarts`  | List of SMARTS patterns to block.        |

**Seed file format by generator**

| Generator  | `smiles_file` format                                          |
|------------|---------------------------------------------------------------|
| Reinvent   | Not required (de-novo generation).                            |
| LibInvent  | 1 scaffold per line, attachment points marked with `*`.       |
| LinkInvent | 2 warhead SMILES per line separated by `\|`.                  |
| Mol2Mol    | 1 compound per line.                                          |
| Pepinvent  | 1 peptide SMILES per line.                                    |


---

## Scoring

Score an existing set of molecules.  No generative model is used.

**Section**: `[parameters]`

| Parameter     | Default | Description                                               |
|---------------|---------|-----------------------------------------------------------|
| `smiles_file` | required | SMILES file; SMILES are read from the first column.      |
| `output_csv`  | —        | Name of the output CSV file (optional).                  |

The scoring setup lives in a `[scoring]` section; see the **Scoring** section
of PARAMS for the full description.


---

## Transfer Learning

Fine-tune an existing model on a target set of SMILES.

**Section**: `[parameters]`

| Parameter                  | Default       | Description                                                                                     |
|----------------------------|---------------|-------------------------------------------------------------------------------------------------|
| `input_model_file`         | required      | Path to the prior model to fine-tune.                                                           |
| `output_model_file`        | required      | Path for the final fine-tuned model.                                                            |
| `smiles_file`              | required      | Training SMILES file (first column is read).                                                    |
| `validation_smiles_file`   | —             | Optional validation SMILES file.                                                                |
| `num_epochs`               | required      | Number of training epochs.                                                                      |
| `batch_size`               | required      | Training mini-batch size.                                                                       |
| `sample_batch_size`        | `100`         | Number of sampled molecules used to compute sample loss statistics.                             |
| `save_every_n_epochs`      | `1`           | Write a checkpoint every N epochs.                                                              |
| `num_refs`                 | `0`           | Number of reference molecules randomly selected for similarity tracking. Set to 0 for large datasets (> 200 molecules). |
| `tb_isim`                  | `false`       | Track iSIM similarity in TensorBoard.                                                           |
| `shuffle_each_epoch`       | `true`        | Shuffle the training set each epoch.                                                            |
| `randomize_smiles`         | `true`        | Randomize atom order in SMILES.                                                                 |
| `standardize_smiles`       | `true`        | Standardize SMILES before training.                                                             |
| `isomeric_smiles`          | `false`       | Generate isomeric SMILES.                                                                       |
| `max_sequence_length`      | `128`         | Maximum token sequence length.                                                                  |
| `n_cpus`                   | `1`           | Number of CPUs used for pair generation (transformer models).                                   |
| `ranking_loss_penalty`     | `false`       | Apply ranking loss penalty (transformer models).                                                |

**Similarity pairs** `[parameters.pairs]` — Mol2Mol / transformer models only:

| Parameter         | Default | Description                                   |
|-------------------|---------|-----------------------------------------------|
| `type`            | `"tanimoto"` | Similarity metric; only `"tanimoto"` is supported. |
| `upper_threshold` | `1.0`   | Maximum Tanimoto similarity for pairing.      |
| `lower_threshold` | `0.7`   | Minimum Tanimoto similarity for pairing.      |
| `min_cardinality` | `1`     | Minimum number of pairs per reference SMILES. |
| `max_cardinality` | `199`   | Maximum number of pairs per reference SMILES. |

**Optional scheduler section** `[scheduler]`: pass a dict of PyTorch LR-scheduler arguments.


---

## Staged Learning

Run reinforcement learning (RL) or curriculum learning (CL).  CL is simply a
multi-stage RL run.  Stages are defined as a TOML list of tables `[[stage]]`.

**Section**: `[parameters]`

| Parameter            | Default         | Description                                                                                                    |
|----------------------|-----------------|----------------------------------------------------------------------------------------------------------------|
| `prior_file`         | required        | Path to the prior (reference) model file.                                                                      |
| `agent_file`         | required        | Path to the agent model file; replace with a checkpoint file to continue a previous run.                       |
| `smiles_file`        | —               | Seed SMILES file for LibInvent, LinkInvent, Mol2Mol and Pepinvent.                                             |
| `summary_csv_prefix` | `"summary"`     | Prefix for the output CSV filename.                                                                            |
| `use_checkpoint`     | `false`         | If `true`, load the diversity filter state from `agent_file` when it is a checkpoint.                         |
| `purge_memories`     | `true`          | If `true`, reset all diversity filter memories after each stage.                                               |
| `batch_size`         | `100`           | Number of molecules generated per RL step.                                                                     |
| `unique_sequences`   | `false`         | If `true`, deduplicate raw token sequences in each step (backward-compatibility option).                       |
| `randomize_smiles`   | `true`          | If `true`, randomize atom order in input SMILES.                                                               |
| `isomeric_smiles`    | `false`         | If `true`, generate isomeric SMILES (transformer models always use isomeric SMILES).                           |
| `sample_strategy`    | `"multinomial"` | Transformer models only: `"multinomial"` or `"beamsearch"`.                                                    |
| `distance_threshold` | `99999`         | Transformer models only: distance threshold for sequence filtering.                                            |
| `temperature`        | `1.0`           | Sampling temperature.                                                                                          |
| `tb_isim`            | `false`         | Track iSIM similarity of generated SMILES vs. all previously generated SMILES in TensorBoard.                  |

**Learning strategy** `[learning_strategy]`:

| Parameter | Default    | Description                                                                 |
|-----------|------------|-----------------------------------------------------------------------------|
| `type`    | `"dap"`    | Reward strategy. Only `"dap"` (Direct Augmented Prior) is currently supported. |
| `sigma`   | `128`      | Controls how dominant the score is in the reward function.                  |
| `rate`    | `0.0001`   | Adam optimizer learning rate.                                               |

**Diversity filter** `[diversity_filter]` — optional, applies globally across all stages:

| Parameter            | Default                       | Description                                                                                         |
|----------------------|-------------------------------|-----------------------------------------------------------------------------------------------------|
| `type`               | required                      | `"IdenticalMurckoScaffold"`, `"IdenticalTopologicalScaffold"`, `"ScaffoldSimilarity"` or `"PenalizeSameSmiles"`. |
| `bucket_size`        | `25`                          | Maximum number of molecules per scaffold bucket before the score is penalized.                      |
| `minscore`           | `0.4`                         | Only memorize a molecule if its score meets this threshold.                                          |
| `minsimilarity`      | `0.4`                         | Minimum scaffold similarity threshold (used by `ScaffoldSimilarity` only).                          |
| `penalty_multiplier` | `0.5`                         | Penalty factor applied per identical SMILES (used by `PenalizeSameSmiles` only).                    |

**Intrinsic penalty** `[intrinsic_penalty]` — optional, overridden by a global diversity filter:

| Parameter          | Default  | Description                                                                                              |
|--------------------|----------|----------------------------------------------------------------------------------------------------------|
| `type`             | required | Currently only `"IdenticalMurckoScaffoldRND"` is supported.                                             |
| `penalty_function` | required | `"Step"`, `"Sigmoid"`, `"Linear"`, `"Tanh"` or `"Erf"`.                                                 |
| `bucket_size`      | `25`     | Maximum number of molecules per scaffold bucket.                                                         |
| `minscore`         | `0.4`    | Score threshold for memorization.                                                                        |
| `learning_rate`    | `0.0001` | Learning rate for the RND prediction network (only used by `IdenticalMurckoScaffoldRND`).               |

**Inception** `[inception]` — optional:

| Parameter     | Default | Description                                                      |
|---------------|---------|------------------------------------------------------------------|
| `smiles_file` | —       | SMILES file with "good" molecules for initial guidance.          |
| `memory_size` | `50`    | Total number of SMILES held in inception memory.                 |
| `sample_size` | `10`    | Number of SMILES randomly sampled from memory each step.         |

**Stage definition** `[[stage]]` — repeat for each stage (must be a TOML array of tables):

| Parameter          | Default  | Description                                                                                           |
|--------------------|----------|-------------------------------------------------------------------------------------------------------|
| `max_steps`        | required | Maximum number of RL optimization steps; when reached **all** stages terminate.                       |
| `max_score`        | `1.0`    | Terminate the stage early when the mean score exceeds this value.                                     |
| `min_steps`        | `50`     | Minimum steps to run before checking the early-termination criterion.                                 |
| `termination`      | `"simple"` | Termination criterion; only `"simple"` is currently supported.                                      |
| `chkpt_file`       | —        | Filename for the checkpoint written at stage end or on Ctrl-C; can be reused as `agent_file`.         |
| `[stage.scoring]`  | required | Scoring configuration for this stage (same structure as the top-level scoring section).               |
| `[stage.diversity_filter]` | — | Per-stage diversity filter; a global `[diversity_filter]` always takes precedence.            |

A stage scoring section can load its setup from an external file:

```toml
[stage.scoring]
type = "geometric_mean"
filename = "stage2_scoring.toml"  # file type inferred from extension
```


---

## Enumeration

Enumerate molecular variants (e.g. peptides) and score them.

**Section**: `[parameters]`

| Parameter                  | Description                                                          |
|----------------------------|----------------------------------------------------------------------|
| `smiles_file`              | SMILES file with template molecules (1 per line).                    |
| `amino_acid_library`       | CSV file with amino acid definitions.                                |
| `amino_acid_name_column`   | Column name in the library CSV containing amino acid names.          |
| `smiles_column`            | Column name in the library CSV containing amino acid SMILES.         |
| `batch_size`               | Number of molecules to process per batch.                            |
| `output_csv`               | Output CSV file name (optional).                                     |

The scoring setup lives in a `[scoring]` section; see the **Scoring** section below.


---

## Data Pipeline

Pre-process and filter a SMILES dataset.  Invoked as a separate entry point
(`reinvent.datapipeline`), not via `run_type`.

| Parameter            | Default   | Description                                                     |
|----------------------|-----------|-----------------------------------------------------------------|
| `input_csv_file`     | required  | Input data file (CSV or similar delimited format).             |
| `smiles_column`      | required  | Column header that contains the SMILES.                        |
| `separator`          | `"\t"`    | Column delimiter.                                               |
| `output_smiles_file` | required  | Output file path for the processed SMILES.                     |
| `num_procs`          | `1`       | Number of parallel worker processes.                            |
| `chunk_size`         | `500`     | Number of molecules processed per chunk.                        |

**Filter section** `[filter]`:

| Parameter               | Default              | Description                                                                    |
|-------------------------|----------------------|--------------------------------------------------------------------------------|
| `elements`              | required             | List of allowed element symbols in addition to organic set (e.g. `["B","P"]`). |
| `transforms`            | `["standard"]`       | List of standardization transforms to apply.                                   |
| `min_heavy_atoms`       | `2`                  | Minimum number of heavy atoms.                                                 |
| `max_heavy_atoms`       | `90`                 | Maximum number of heavy atoms.                                                 |
| `max_mol_weight`        | `1200.0`             | Maximum molecular weight (Da).                                                 |
| `min_carbons`           | `2`                  | Minimum number of carbon atoms.                                                |
| `max_num_rings`         | `12`                 | Maximum number of rings.                                                       |
| `max_ring_size`         | `7`                  | Maximum ring size (number of atoms).                                           |
| `keep_stereo`           | `true`               | Preserve stereochemistry.                                                      |
| `keep_isotope_molecules`| `true`               | Keep isotopically-labelled molecules.                                          |
| `uncharge`              | `true`               | Remove formal charges.                                                         |
| `kekulize`              | `false`              | Convert aromatic bonds to Kekulé form.                                         |
| `randomize_smiles`      | `false`              | Randomize atom order in output SMILES.                                         |
| `report_errors`         | `false`              | Log processing errors.                                                         |


---

## Scoring configuration

All run modes that score molecules use the same scoring block structure.

**Top-level scoring section** (`[scoring]` in most run modes, `[stage.scoring]` inside staged learning):

| Parameter  | Default | Description                                                                                            |
|------------|---------|--------------------------------------------------------------------------------------------------------|
| `type`     | required | Aggregation function: `"arithmetic_mean"` / `"custom_sum"` or `"geometric_mean"` / `"custom_product"`. |
| `parallel` | `1`      | Number of CPU cores to use for component computation (maximum 40).                                     |
| `use_pumas`| `false`  | Use the alternative PUMAS desirability transform library instead of the built-in transforms.  Requires the optional `pumas` extra: `pip install reinvent[pumas]`. |
| `filename` | —        | Load the scoring setup from this external file (TOML or JSON).                                         |
| `filetype` | —        | File format when `filename` is set: `"toml"` or `"json"` (optional, filetype inferred from suffix).    |

**Component list** `[[scoring.component]]` — one block per component type.  Each component block
contains an `endpoint` list and optionally component-level `params`.

```toml
[[scoring.component]]
[scoring.component.MolecularWeight]

[[scoring.component.MolecularWeight.endpoint]]
name = "MW"           # user-chosen label for output columns
weight = 0.5          # relative weight (normalized across all components)

params.smiles = [...]   # component-specific parameters

transform.type = "double_sigmoid"  # optional transform
transform.high = 500.0
transform.low  = 200.0
```

**Component roles**

| Role    | Description                                                                        |
|---------|------------------------------------------------------------------------------------|
| scorer  | Default. Score is included in the weighted aggregation.                            |
| filter  | Applied globally before aggregation; a failing molecule scores zero. No weight.    |
| penalty | Included in the weighted aggregation but intended to penalize unwanted features.   |