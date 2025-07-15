#!/bin/bash

source /home/me/miniconda/bin/activate aizynthfinder

aizynth_config="config.yml"  # CAZP generated AiZynthFinder configuration
nproc=8

# the SMILES are reveived via stdin
cp /dev/stdin input.smi

aizynthcli --smiles input.smi \
    --config "$aizynth_config" \
    --output output.json.gz \
    --nproc "$nproc" >> aizynth-stdout-stderr.txt 2>&1

# write the CAZP JSON files to stdout
zcat output.json.gz
