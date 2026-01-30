
import os
import pandas as pd
import reinvent

def run_reinforcement_learning(args, wd):

    global_parameters = f"""
        run_type = "staged_learning"
        device = "cuda:0"
        tb_logdir = "{wd}/tensorboard"
        json_out_config = "{wd}/_rl.json"
    """

    # ### Parameters
    #
    # Here we specify the model files, the prefix for the output CSV summary file and the batch size for sampling and stochastic gradient descent (SGD).  The batch size is often given in 2^N but there is in now way required.  Typically batch sizes are between 50 and 150.  Batch size effects on SGD and so also the learning rate.  Some experimentation may be required to adjust this but keep in mind that, say, raising the total score as fast as possible is not necessarily the best choice as this may hamper exploration.

    # +
    prior_filename = os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior")
    agent_filename = prior_filename

    parameters = f"""
    [parameters]

    prior_file = "{prior_filename}"
    agent_file = "{agent_filename}"
    summary_csv_prefix = "{wd}/rl"

    batch_size = 32

    use_checkpoint = false
    """
    # -

    # ### Reinforcement Learning strategy

    learning_strategy = """
    [learning_strategy]

    type = "dap"
    sigma = 128
    rate = 0.0001
    """

    # ###  Stage setup
    #
    # Here we only use a single stage. The aim of this stage is to create an agent which is highly likely to generate "drug-like" molecules (as per QED and Custom Alerts) with no stereocentres
    #
    # The stage will terminate when a maximum number of 300 steps is reached.  Termination could occur earlier when the maximum score of 1.0 is exceeded but this is very unlikely to occur.  A checkpoint file is written out which can be used as the agent in a subsequent stage.
    #
    # The scoring function is a weighted product of all the scoring components: QED and number of sterecentres.  The latter is used here to avoid stereocentres as they are not supported by the Reinvent prior.  Zero stereocentres aids in downstream 3D task to avoid having to carry out stereocentre enumeration.  Custom alerts is a filter which filters out (scores as zero) all generated compounds which match one of the SMARTS patterns.  Number of sterecentres uses a transformation function to ensure the component score is between 0 and 1.

    stages = f"""
    [[stage]]

    max_score = 1.0
    max_steps = 300

    chkpt_file = '{wd}/checkpoints/rl.chkpt'

    [stage.scoring]
    type = "geometric_mean"

    [[stage.scoring.component]]
    [stage.scoring.component.custom_alerts]

    [[stage.scoring.component.custom_alerts.endpoint]]
    name = "Alerts"

    params.smarts = [
        "[*;r8]",
        "[*;r9]",
        "[*;r10]",
        "[*;r11]",
        "[*;r12]",
        "[*;r13]",
        "[*;r14]",
        "[*;r15]",
        "[*;r16]",
        "[*;r17]",
        "[#8][#8]",
        "[#6;+]",
        "[#16][#16]",
        "[#7;!n][S;!$(S(=O)=O)]",
        "[#7;!n][#7;!n]",
        "C#C",
        "C(=[O,S])[O,S]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
        "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
        "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
        "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
    ]

    [[stage.scoring.component]]
    [stage.scoring.component.QED]

    [[stage.scoring.component.QED.endpoint]]
    name = "QED"
    weight = 0.6


    [[stage.scoring.component]]
    [stage.scoring.component.NumAtomStereoCenters]

    [[stage.scoring.component.NumAtomStereoCenters.endpoint]]
    name = "Stereo"
    weight = 0.4

    transform.type = "left_step"
    transform.low = 0
    """

    config = global_parameters + parameters + learning_strategy + stages

    toml_config_filename = f"{wd}/rl.toml"

    with open(toml_config_filename, "w") as tf:
        tf.write(config)

    import subprocess
    print("Running reinvent")
    try:   
        # Define the command
        command = f"reinvent -l {wd}/rl.log {toml_config_filename}"

        # Run the command
        process = subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

    print("Finished running reinvent")
    
    if args.statistics:
        csv_file = os.path.join(wd, "rl_1.csv")
        df = pd.read_csv(csv_file)
        total_smilies = len(df)
        invalids = df[df["SMILES_state"] == 0]
        total_invalid_smilies = len(invalids)
        duplicates = df[df["SMILES_state"] == 2]
        total_batch_duplicate_smilies = len(duplicates)
        all_duplicates = df[df.duplicated(subset=["SMILES"])]
        total_duplicate_smilies = len(all_duplicates)

        print(
            f"Total number of SMILES generated: {total_smilies}\n"
            f"Total number of invalid SMILES generated: {total_invalid_smilies}\n"
            f"Total number of batch duplicate SMILES generated: {total_batch_duplicate_smilies}\n"
            f"Total number of duplicate SMILES generated: {total_duplicate_smilies}"
        )

