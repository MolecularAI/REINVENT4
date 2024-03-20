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

    hash_id: str | None # hash of the entire state dictionary minus the hash fields
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


def update_model_data(save_dict: dict) -> dict:
    """Compute the hash for the model data

    :param save_dict: the model description
    :returns: updated save dict
    """

    save_dict = copy.deepcopy(save_dict)
    save_dict = {k: v for k, v in sorted(save_dict.items())}

    metadata = save_dict["metadata"]

    # check if metadata does not exist, do nothing
    if metadata is None:
        return save_dict

    # do not hash the hash and its format itself
    metadata.hash_id = None
    metadata.hash_id_format = None

    # FIXME: what if this gets "too long"?
    metadata.updates.append((time.time(),))

    if "decorator" in save_dict:  # Libinvent
        ref = save_dict["decorator"]["state"]
    elif "network_state" in save_dict:  # Linkinvnet, Mol2Mol
        ref = save_dict["network_state"]
    else:  # Reinvent
        ref = save_dict["network"]

    for k in ref:
        ref[k] = ref[k].cpu().numpy()

    data = pickle.dumps(save_dict)

    metadata.hash_id = xxhash.xxh3_128_hexdigest(data)
    metadata.hash_id_format = HASH_FORMAT

    return save_dict


def check_valid_hash(save_dict: dict) -> bool:
    """Check the hash of the model data

    :param save_dict: the model description
    :returns: whether hash is valid
    """

    save_dict = copy.deepcopy(save_dict)
    save_dict = {k: v for k, v in sorted(save_dict.items())}

    metadata = save_dict["metadata"]

    curr_hash_id = metadata.hash_id
    curr_hash_id_format = metadata.hash_id_format

    # do not hash the hash and its format itself
    metadata.hash_id = None
    metadata.hash_id_format = None

    if "decorator" in save_dict:  # Libinvent
        ptr = save_dict["decorator"]["state"]
    elif "network_state" in save_dict:  # Linkinvnet, Mol2Mol
        ptr = save_dict["network_state"]
    else:  # Reinvent
        ptr = save_dict["network"]

    for k in ptr:
        ptr[k] = ptr[k].cpu().numpy()

    data = pickle.dumps(save_dict)

    check_hash_id = xxhash.xxh3_128_hexdigest(data)

    metadata.hash_id = curr_hash_id
    metadata.hash_id_format = curr_hash_id_format

    return curr_hash_id == check_hash_id
