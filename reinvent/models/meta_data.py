"""Meta data definition for model files"""

from __future__ import annotations

__all__ = ["ModelMetaData", "update_model_data", "check_valid_hash"]
from dataclasses import dataclass, field, asdict
import time
import copy
import pickle
import xxhash
import platform
from uuid import UUID
from typing import List, Tuple

HASH_FORMAT = f"xxhash.xxh3_128_hex {xxhash.VERSION}"
MODEL_ID_FORMAT = f"uuid.uuid4 {platform.python_version()}"


@dataclass
class ModelMetaData:
    """Meta data for model files to support data provenance and lineage

    model_id, creation_date should only be created once after a model has been
    freshly trained
    hash_id, hash_id_format need to be excluded or use default for computation
    """

    hash_id: str | None  # hash of the entire state dictionary minus the hash fields
    hash_id_format: str
    model_id: UUID | str  # uuid.uuid4() or https://pypi.org/project/uuid6/
    origina_data_source: str  # e.g. "ChEMBL 33" or "PubChem 2023-11-23"
    creation_date: float  # epoch e.g. time.time()
    date_format: str = "UNIX epoch"
    model_id_format: str = MODEL_ID_FORMAT
    updates: List[Tuple] = field(default_factory=list)  # list of epochs
    comments: List[str] = field(default_factory=list)  # arbitrary comment e.g. user annotation

    def as_dict(self):
        return asdict(self)


def update_model_data(save_dict: dict, comment: str = "", write_update: bool = True) -> dict:
    """Compute the hash for the model data

    Works on a copy of save_dict.
    NOTE: This will modify the original save_dict as it converts the tensors
          to numpy arrays. Use the returned save_dict!

    :param save_dict: the model description
    :param comment: the comment for current update
    :param write_update: whether to write the update or not
    :returns: updated save dict with the metadata as dict
    """

    # copy and sort
    save_dict = {k: v for k, v in sorted(save_dict.items())}
    metadata = save_dict["metadata"]

    if not isinstance(save_dict["metadata"], dict):
        metadata = metadata.as_dict()

    # FIXME: what if this gets "too long"?
    if write_update:
        metadata["updates"].append(time.time())

    if comment:
        metadata["comments"].append(comment)

    # do not hash the hash itself and its format
    metadata["hash_id"] = None
    metadata["hash_id_format"] = None

    ref = _get_network(save_dict)
    network = copy.deepcopy(ref)  # keep original tensors

    # convert to numpy arrays to avoid hashing on torch.tensor metadata
    # only needed for hashing, will copy back tensors further down
    for k in sorted(ref.keys()):
        ref[k] = ref[k].cpu().numpy()

    save_dict["metadata"] = metadata

    data = pickle.dumps(save_dict)

    metadata["hash_id"] = xxhash.xxh3_128_hexdigest(data)
    metadata["hash_id_format"] = HASH_FORMAT

    return _set_network(save_dict, network)


def check_valid_hash(
    save_dict: dict,
) -> bool:
    """Check the hash of the model data

    Works on a copy of save_dict.  save_dict should not be used any further
    because the parameters, etc. are in numpy format.

    :param save_dict: the model description, metadata expected as dict
    :returns: whether hash is valid
    """

    save_dict = copy.deepcopy(save_dict)  # avoid problems modifying the original
    save_dict = {k: v for k, v in sorted(save_dict.items())}
    metadata = save_dict["metadata"]

    if isinstance(save_dict["metadata"], dict):  # new models
        curr_hash_id = metadata["hash_id"]
        metadata["hash_id"] = None
        metadata["hash_id_format"] = None
    else:  # ModelMetaData for legacy models
        curr_hash_id = metadata.hash_id
        metadata.hash_id = None
        metadata.hash_id_format = None

    ref = _get_network(save_dict)

    for k in sorted(ref.keys()):
        ref[k] = ref[k].cpu().numpy()

    data = pickle.dumps(save_dict)
    check_hash_id = xxhash.xxh3_128_hexdigest(data)

    return curr_hash_id == check_hash_id


def _get_network(save_dict: dict) -> dict:
    if "decorator" in save_dict:  # Libinvent
        ref = save_dict["decorator"]["state"]
    elif "network_state" in save_dict:  # Linkinvent, Mol2Mol
        ref = save_dict["network_state"]
    else:  # Reinvent
        ref = save_dict["network"]

    return ref


def _set_network(save_dict, network):
    if "decorator" in save_dict:  # Libinvent
        save_dict["decorator"]["state"] = network
    elif "network_state" in save_dict:  # Linkinvent, Mol2Mol
        save_dict["network_state"] = network
    else:  # Reinvent
        save_dict["network"] = network

    return save_dict
