import numpy as np

from reinvent.models.model_factory.sample_batch import SmilesState, SampleBatch


def common_report(sampled: SampleBatch, **kwargs):
    valid_mask = np.where(
        (sampled.states == SmilesState.VALID) | (sampled.states == SmilesState.DUPLICATE),
        True,
        False,
    )
    unique_mask = np.where(sampled.states == SmilesState.VALID, True, False)

    fraction_valid_smiles = sum(valid_mask) / len(sampled.states)
    fraction_unique_molecules = sum(unique_mask) / len(sampled.states)

    additional_report = {}

    if "Tanimoto" in kwargs.keys():
        tanimoto_scores = kwargs["Tanimoto"]
        nlls = sampled.nlls.cpu().detach().numpy()

        additional_report["Tanimoto_valid"] = np.array(tanimoto_scores)[valid_mask].tolist()
        additional_report["Tanimoto_unique"] = np.array(tanimoto_scores)[unique_mask].tolist()
        additional_report["Output_likelihood_valid"] = nlls[valid_mask].tolist()
        additional_report["Output_likelihood_unique"] = nlls[unique_mask].tolist()

    return fraction_valid_smiles, fraction_unique_molecules, additional_report
