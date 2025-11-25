#!/usr/bin/env bash

aizynth_config="config_synthsense.yml"  # Default.
nproc_line=""  # Default to empty line, i.e. do not include --nproc.

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) aizynth_config="$2"; shift ;;
        --nproc) nproc_line=" --nproc $2 "; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f ${aizynth_config} ]; then
    >&2 echo "Config file ${aizynth_config} not found! Provide config with --config option, or put config.yml in current directory."
    exit 1
fi

cp /dev/stdin input.smi 

# activate aizynthfinder virtual env

export PATH="miniforge3/envs/aizynth/bin:$PATH"
source "miniforge3/bin/activate" "aizynth"


# Log the number of input SMILES
wc -l input.smi >> aizynth-stdout-stderr.txt

# aizynthcli --smiles input.smi --config ${aizynth_config} --output output.json.gz ${nproc_line} >>aizynth-stdout-stderr.txt 2>&1
aizynthcli --smiles input.smi --config ${aizynth_config} --output output.json.gz --nproc 12 >>aizynth-stdout-stderr.txt 2>&1

zcat output.json.gz
