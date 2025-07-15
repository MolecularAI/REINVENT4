#!/bin/bash
#
# Simple run script for running AiZynthFinder via slurm.


source /home/me/miniconda/bin/activate aizynthfinder

worker="worker.sh"
config="${PWD}/config.yml"  # CAZP generated AiZynthFinder configuration
nmols=8

# split the SMILES from stdin into batches
split --numeric-suffixes=1 --additional-suffix=_inp.smi \
    --lines=${nmols} --suffix-length=2 -


# write out the slurm script
cat > "$worker" << _EOF
#!/bin/bash

step="\$1"

mkdir "worker_\$step"
cd "worker_\$step"

aizynthcli --smiles "${PWD}/x0\${step}_inp.smi" \
    --output "${PWD}/\${step}_out.json.gz" \
    --config "$config" --nproc "$nmols" \
    > aizynth-stdout-stderr.txt 2>&1
_EOF

chmod u+x "$worker"

nodes="--nodes=1 --ntasks-per-node=1 --cpus-per-task=$nmols"

# Run 2 slurm jobs on 2 nodes
srun $nodes --output="s1.out" "$worker" 1 &
srun $nodes --output="s2.out" "$worker" 2 &

wait

# comcatenate all JSON outputs
cat_aizynth_output --files *_out.json.gz --output output.json.gz \
    >> cat_aizynth_output-stdout-stderr.txt 2>&1

# write concatenated JSON to stdout
zcat output.json.gz

