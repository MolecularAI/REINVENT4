from datasets import load_dataset
import os

def prep_data(args):
    print("Downloading tack data")
    tack_ds = load_dataset("ailab-bio/TACK")
    tack_ds_train = tack_ds["train"].to_pandas()

    base_path = os.getcwd() + "/" + args.data_folder

    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    TL_train_filename = f"{base_path}/tack_train.smi"
    TL_validation_filename = f"{base_path}/tack_validation.smi"
    tack_smiles = tack_ds_train["SMILES"]

    #FIXME: temporary solution to filter out unallowed tokens for reinvent.prior
    reinvent_prior_allowed_tokens = {')', 'S', '^', '2', 'O', '%10', '4', '=', 'C', '1', '9', '6', 's', '[nH]', '5', 'Br', 'o', '7', '(', '[S+]', 'n', '-', '8', 'N', '[N+]', 'F', '3', '[N-]', 'c', '[O-]', '[n+]', 'Cl', '#', '$'}


    tack_smiles = tack_smiles[tack_smiles.apply(lambda x: all(token in reinvent_prior_allowed_tokens for token in x))]

    n_head = int(0.8 * len(tack_smiles))  # 80% of the data for training
    n_tail = len(tack_smiles) - n_head
    print(f"number of molecules for: training={n_head}, validation={n_tail}")

    train, validation = tack_smiles.head(n_head), tack_smiles.tail(n_tail)

    train.to_csv(TL_train_filename, sep="\t", index=False, header=False)
    validation.to_csv(TL_validation_filename, sep="\t", index=False, header=False)
    print(f"Finished writing curated data to folder {base_path}")

    print("Downloading synthetic data")

    synthetic_ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered")

    for split in synthetic_ds:
        synthetic_ds[split] = synthetic_ds[split].remove_columns("labels")
        # write only first 200 000 molecules to file to limit size
        #synthetic_ds[split] = synthetic_ds[split].select(range(min(500000, len(synthetic_ds[split]))))

        synthetic_ds[split] = synthetic_ds[split].filter(lambda x: all(token in reinvent_prior_allowed_tokens for token in x["text"]))

        print(f"After filtering, {split} split has {len(synthetic_ds[split])} molecules")

        synthetic_ds[split].to_csv(base_path + f"/synthetic_{split}.smi", header=False)    

    print(f"Finished writing synthetic data to {base_path}")
