# Autodock GPU
mk_prepare_receptor.py \
  --read_pdb 1DB5_apo.pdb \
  --output_basename 1DB5 \
  --write_pdbqt --write_gpf \
  --box_enveloping 6IN.sdf --padding 5

autogrid4 -p 1DB5.gpf -l 1DB5.log
