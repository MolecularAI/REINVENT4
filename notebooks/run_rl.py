import os
import pandas as pd
import subprocess
import re

def get_scorer(is_stage_2_RL):
    # QED prob not good since lower desirability for heavier molecules (kind of follows Ro5)
    if not is_stage_2_RL:
        return """
            [stage.scoring]
            type = "arithmetic_mean"

            [[stage.scoring.component]]

            [stage.scoring.component.MolecularWeight]

            [[stage.scoring.component.MolecularWeight.endpoint]]

            name = "MW"

            weight = 1.0
            transform.type = "Double_Sigmoid"

            transform.low = 700.0

            transform.high = 1200.0


            [[stage.scoring.component]]
            [stage.scoring.component.QED]
            [[stage.scoring.component.QED.endpoint]]
            name = "QED"
            weight = 2.0

            [[stage.scoring.component]]
            [stage.scoring.component.SlogP]
            [[stage.scoring.component.SlogP.endpoint]]
            name = "LogP_Limit"
            weight = 0.5
            transform.type = "Double_Sigmoid"
            transform.low = 3.0
            transform.high = 7.0
            transform.coef_div = 0.5

            [[stage.scoring.component]]

            [stage.scoring.component.SAScore]

            [[stage.scoring.component.SAScore.endpoint]]

            name = "SAscore"

            weight = 1.00

            transform.type = "Right_Step"

            transform.high = 6.0
            transform.low = 4.0


            [stage.diversity_filter]
            type = "IdenticalMurckoScaffold"
            bucket_size = 25
            minscore = 0.4
            minsimilarity = 0.4
        """
        #return
        """ [[stage.scoring.component]]
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
    return """
            [stage.scoring]
            type = "geometric_mean"

            [[stage.scoring.component]]

            [stage.scoring.component.TPSA]

            [[stage.scoring.component.TPSA.endpoint]]
            name = "ScoringComponent"

            weight = 1.5


            [[stage.scoring.component]]

            [stage.scoring.component.MolecularWeight]

            [[stage.scoring.component.MolecularWeight.endpoint]]

            name = "MW"

            weight = 1.0
            transform.type = "Double_Sigmoid"

            transform.low = 900.0

            transform.high = 1100.0


            [[stage.scoring.component]]

            [stage.scoring.component.NumRotBond]

            [[stage.scoring.component.NumRotBond.endpoint]]

            name = "NumRotBond"

            weight = 1.00
            transform.type = "Double_Sigmoid"

            transform.low = 10.0

            transform.high = 25.0


            [stage.diversity_filter]
            type = "IdenticalMurckoScaffold"
            bucket_size = 25
            minscore = 0.4
            minsimilarity = 0.4
        """

    #return 
    """
        [[stage.scoring.component]]
        [[stage.scoring.component.AtomCount.endpoint]]
        name = "AtomCount"
        params.target = "O"
        weight = 1.0

        [[stage.scoring.component.AtomCount.endpoint]]
        name = "AtomCount"
        params.target = "C"
        weight = 1.0

        """




def run_reinforcement_learning(args, wd, is_stage_2_RL=False, min_steps=200, max_steps=1000):

    checkpoints_wd = f"{wd}/checkpoints"
    if not os.path.isdir(checkpoints_wd):
        os.mkdir(checkpoints_wd)
 
    global_parameters = f"""
        run_type = "staged_learning"
        device = "cuda:0"
        tb_logdir = "{wd}/tb"
        json_out_config = "{wd}/_rl.json"
    """

    prior_filename = args.prior
    agent_filename = prior_filename

    parameters = f"""
    [parameters]

    prior_file = "{prior_filename}"
    agent_file = "{agent_filename}"
    summary_csv_prefix = "{wd}/rl"

    batch_size = 256

    use_checkpoint = false
    """

    learning_strategy = """
    [learning_strategy]

    type = "dap"
    sigma = 128
    rate = 0.0001
    """

    stage_config = f"""
    [[stage]]

    max_steps = {max_steps}
    min_steps = {min_steps}

    chkpt_file = '{wd}/checkpoints/rl.chkpt'

    """

    if is_stage_2_RL:
        
        agent_filename = ""
        tl_checkpoint = args.tl_checkpoint
        if str.lower(args.run) == "both" and not tl_checkpoint:
            TL_checkpoint_path = f"{wd[:-11]}/Stage_3_RL/checkpoints"
            checkpoint_files = os.listdir(TL_checkpoint_path)
            agent_filename = max(checkpoint_files, key=lambda d: os.path.getmtime(os.path.join(TL_checkpoint_path, d)))

        elif not tl_checkpoint:
            raise Exception("""
                    Cannot run stage 2 reinforcement learning without a TL model checkpoint. 
                    Define using '--tl-checkpoint=$PATH'
                """)
        
        else: 
            agent_filename = tl_checkpoint

        stage_config = re.sub("stage1", f"stage2", stage_config)
        stage_config = re.sub("agent_file.*\n", f"agent_file = '{agent_filename}'\n", stage_config)
        stage_config = re.sub("max_steps.*\n", f"max_steps = 500\n", stage_config)

    stage_config += get_scorer(is_stage_2_RL)

    config = global_parameters + parameters + learning_strategy + stage_config

    toml_config_filename = f"{wd}/rl.toml"

    with open(toml_config_filename, "w") as tf:
        tf.write(config)


    print("Running reinvent reinforcement learning")
    try:   
        # Define the command
        command = f"reinvent -l {wd}/rl.log {toml_config_filename}"

        # Run the command
        process = subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

    print("Finished running reinvent reinforcement learning")
    
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

