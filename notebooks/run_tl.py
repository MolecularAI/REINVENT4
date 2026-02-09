import os
import shutil
import reinvent
import subprocess


def run_transfer_learning(args, wd, data_prefix, batch_size=512, num_epochs=5):

    checkpoints_wd = f"{wd}/checkpoints"
    print("Creating TL checkpoints directory at:", checkpoints_wd)
    if not os.path.isdir(checkpoints_wd):
        os.mkdir(checkpoints_wd)

    """#FIXME: change, only uses reinvent.prior temporaryly instead of checkpoint
    prior_filename = os.path.join(wd, "..", "Stage_1_RL/checkpoints", "rl.chkpt")

    # check if prior RL stage 1 checkpoint exists
    if not os.path.isfile(prior_filename):"""
        # fall back to default prior, user selected TL only run
    prior_filename = os.path.abspath(os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior"))        
    
    print("Using prior filename:", prior_filename)

    base_path = os.getcwd() + "/" + args.data_folder

    TL_train_filename = f"{base_path}/{data_prefix}_train.smi"
    TL_validation_filename = f"{base_path}/{data_prefix}_validation.smi"

    # #### TL setup
    tb_logdir = f"{wd}/tb_0"
    output_model_file = f"{wd}/checkpoints/TL_reinvent.model"

    TL_parameters = f"""
    run_type = "transfer_learning"
    device = "cuda:0"
    tb_logdir = "{tb_logdir}"


    [parameters]

    num_epochs = {num_epochs}
    save_every_n_epochs = 2
    batch_size = {batch_size}
    sample_batch_size = 128

    input_model_file = "{prior_filename}" 
    output_model_file = "{output_model_file}"
    smiles_file = "{TL_train_filename}"
    validation_smiles_file = "{TL_validation_filename}"
    standardize_smiles = false
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







