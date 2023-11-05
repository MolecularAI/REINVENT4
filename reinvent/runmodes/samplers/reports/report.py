import numpy as np

from reinvent.models.model_factory.sample_batch import SmilesState, SampleBatch


def report_setup(sampled: SampleBatch, time: int, **kwargs):
    fraction_valid_smiles = 100*len(np.where(sampled.states != SmilesState.INVALID)[0]) / len(sampled.states)
    fraction_unique_molecules = 100*len(np.where(sampled.states == SmilesState.VALID)[0]) / len(sampled.states)

    additional_report = {}
    if 'Tanimoto' in kwargs.keys():
        tanimoto_scores = kwargs['Tanimoto']
        # Log all valid molecules' stat, include duplicated
        additional_report['Tanimoto_valid'] = \
            np.array(tanimoto_scores)[np.where(sampled.states != SmilesState.INVALID)].tolist()
        # Log all unique molecules' stat
        additional_report['Tanimoto_unique'] = \
            np.array(tanimoto_scores)[np.where(sampled.states == SmilesState.VALID)].tolist()
        # Log NLL stat
        additional_report['Output_likelihood_valid'] = \
            sampled.nlls.cpu().detach().numpy()[np.where(sampled.states != SmilesState.INVALID)].tolist()
        additional_report['Output_likelihood_unique'] = \
            sampled.nlls.cpu().detach().numpy()[np.where(sampled.states == SmilesState.VALID)].tolist()

    return fraction_valid_smiles, fraction_unique_molecules, time,  additional_report
