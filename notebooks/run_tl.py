import os
import shutil
import reinvent
from datasets import load_dataset
import subprocess


def run_transfer_learning(args, wd):

    checkpoints_wd = f"{wd}/checkpoints"
    if not os.path.isdir(checkpoints_wd):
        os.mkdir(checkpoints_wd)

    prior_filename = os.path.abspath(os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior"))

    tack_ds = load_dataset("ailab-bio/TACK")

    tack_ds_train = tack_ds["train"].to_pandas()

    TL_train_filename = f"{wd}/tack_train.smi"
    TL_validation_filename = f"{wd}/tack_validation.smi"
    tack_smiles = tack_ds_train["SMILES"]

    #remove smiles containing %11 #FIXME: not supported by reinvent.prior change later
    tack_smiles = tack_smiles[~tack_smiles.str.contains("%11")]
    for smi in tack_smiles:
        if "%11" in smi:
            print("FOUND IT:")
            print(smi)

    n_head = int(0.8 * len(tack_smiles))  # 80% of the data for training
    n_tail = len(tack_smiles) - n_head
    print(f"number of molecules for: training={n_head}, validation={n_tail}")

    train, validation = tack_smiles.head(n_head), tack_smiles.tail(n_tail)

    train.to_csv(TL_train_filename, sep="\t", index=False, header=False)
    validation.to_csv(TL_validation_filename, sep="\t", index=False, header=False)


    # #### TL setup
    #FIXME: change, only uses reinvent.prior temporaryly instead of checkpoint
    tb_logdir = f"{wd}/tb_0"
    output_model_file = f"{wd}/checkpoints/TL_reinvent.model"

    TL_parameters = f"""
    run_type = "transfer_learning"
    device = "cuda:0"
    tb_logdir = "{tb_logdir}"


    [parameters]

    num_epochs = 3
    save_every_n_epochs = 1
    batch_size = 12
    sample_batch_size = 100

    input_model_file = "{prior_filename}" 
    output_model_file = "{output_model_file}"
    smiles_file = "{TL_train_filename}"
    validation_smiles_file = "{TL_validation_filename}"
    standardize_smiles = true
    randomize_smiles = false
    randomize_all_smiles = false
    internal_diversity = true
    """

    # +
    TL_config_filename = f"{wd}/transfer_learning.toml"

    with open(TL_config_filename, "w") as tf:
        tf.write(TL_parameters)
    # -

    # ## Start Transfer Learning
    tl_logfile = f"{wd}/tl.log"

    shutil.rmtree(tb_logdir, ignore_errors=True)

    print("Running reinvent transfer learning")
    try:   
        # Define the command
        command = f"reinvent -l {tl_logfile} {TL_config_filename}"

        # Run the command
        process = subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

    print("Finished running reinvent transfer learning")







