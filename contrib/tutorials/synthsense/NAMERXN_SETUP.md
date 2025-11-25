# NameRxn Setup for SynthSense

This guide explains how to set up **NameRxn** (NextMove Software) to provide reaction class annotations for AiZynthFinder, which are essential for running SynthSense.

## Why NameRxn is Needed

SynthSense rewards rely on **reaction class annotations** to:
- Evaluate synthesis routes based on preferred reaction types
- Track route diversity


## Step 1: Obtain NameRxn

NameRxn is a **commercial tool** from NextMove Software that must be purchased.

## Step 2: Annotate Reaction Templates

AiZynthFinder uses reaction templates. These templates need reaction class annotations from NameRxn.

### 2.1 Download USPTO Template Library

Download the base USPTO template files from Zenodo:

**Source**: https://zenodo.org/records/7341155

You need two files:
- `uspto_template_library.csv` - Full template library with reaction SMILES
- `uspto_unique_templates.csv.gz` - Unique templates (to be annotated)

### 2.2 Install rxnutils

Install [rxnutils](https://molecularai.github.io/reaction_utils/index.html) for pipeline processing:

```bash
pip install reaction-utils
```

### 2.3 Create NameRxn Pipeline Configuration

Create a file called `nm_pipeline.yaml`:

```yaml
namerxn:
    in_column: reaction_smiles
```

This tells rxnutils to run NameRxn on the `reaction_smiles` column.

### 2.4 Run NameRxn Classification Pipeline

Execute the NameRxn pipeline on the template library:

```bash
python -m rxnutils.pipeline.runner \
    --pipeline nm_pipeline.yaml \
    --data uspto_template_library.csv \
    --output uspto_template_library-nm.csv
```

**Requirements**:
- `namerxn` must be in your system PATH
- This will process all reactions in the template library

### 2.5 Update Unique Templates with Classifications

Use this Python script to transfer classifications to the unique templates file:

```python
import pandas as pd
from collections import Counter

template_lib = pd.read_csv(
    "uspto_template_library-nm.csv", 
    sep="\t", 
    usecols=["template_hash", "NMC"]
)

unique_templates = pd.read_csv(
    "uspto_unique_templates.csv.gz", 
    index_col=0, 
    sep="\t"
)

hash_to_class = template_lib.groupby("template_hash")["NMC"].apply(
    lambda x: Counter([
        cls for val in x for cls in val.split(";")
    ]).most_common(1)[0][0]
).to_dict()

new_classification = unique_templates.template_hash.apply(
    lambda x: hash_to_class[x]
)
unique_templates["classification"] = new_classification

unique_templates.to_csv(
    "uspto_unique_templates_with_classes.csv.gz", 
    sep="\t"
)
```

## Step 3: Configure AiZynthFinder

### 3.1 Update AiZynthFinder Config

Modify your AiZynthFinder configuration (`configs/config_synthsense.yml`) to use the annotated templates:

```yaml
expansion:
  uspto:
    - /path/to/uspto_model.onnx
    - /path/to/uspto_unique_templates_with_classes.csv.gz  # Your annotated file
```

### 3.3 Configure Reaction Class Scorers

In your SynthSense config (`configs/config_synthsense.yml`), specify which reaction classes to favor:

```yaml
scorer:
  ReactionClassMembershipScorer:
    in_set_score: 1.0
    not_in_set_score: 0.1
    reaction_class_set: [
      "3.1.1",  # Bromo Suzuki coupling
      "2.1.1",  # Amide Schotten-Baumann
      "1.3.1",  # Bromo Buchwald-Hartwig amination
      # ... add your preferred reaction classes
    ]
```

