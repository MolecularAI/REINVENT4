import numpy as np
import torch

from reinvent.runmodes.RL import Learning


def get_distance_to_prior(likelihood, distance_threshold: float) -> np.ndarray:
    # FIXME: the datatype should not be variable
    if isinstance(likelihood, torch.Tensor):
        ones = torch.ones_like(likelihood, requires_grad=False)
        mask = torch.where(likelihood < distance_threshold, ones, distance_threshold / likelihood)
        mask = mask.cpu().numpy()
    else:
        ones = np.ones_like(likelihood)
        mask = np.where(likelihood < distance_threshold, ones, distance_threshold / likelihood)

    return mask


def score(learning: Learning):
    """Compute the score for the SMILES strings."""
    prior_nll = learning.prior.likelihood_smiles(learning.sampled).likelihood
    distance_penalties = get_distance_to_prior(prior_nll, learning.distance_threshold)

    results = learning.scoring_function(
        learning.sampled.smilies, learning.invalid_mask, learning.duplicate_mask
    )

    results.total_scores *= distance_penalties

    return results
