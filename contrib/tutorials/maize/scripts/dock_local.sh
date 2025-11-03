# Run Gnina with local optimization

receptor="$1"
in_sdf="$2"
out_sdf="$3"

gnina.1.3.2.cuda12.8 \
    --receptor "$receptor" \
    --ligand "$in_sdf" \
    --out "$out_sdf" \
    --local_only \
    --minimize \
    --cnn_scoring rescore
    #--scoring vinardo
