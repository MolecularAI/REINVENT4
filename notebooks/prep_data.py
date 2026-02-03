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
    print(f"Finished writing curated data to folder {base_path}")

    print("Downloading synthetic data")

    synthetic_ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered")

    for split in synthetic_ds:
        synthetic_ds[split] = synthetic_ds[split].remove_columns("labels")
        synthetic_ds[split].to_csv(base_path + f"/synthetic_{split}.smi", header=False)

    print(f"Finished writing synthetic data to {base_path}")
