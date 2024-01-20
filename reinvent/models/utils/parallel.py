from concurrent.futures import ProcessPoolExecutor
from typing import Callable
import multiprocessing as mp
import os

import numpy as np


def parallel(n_cpus: int, shared_kwargs: dict, func: Callable, *args, **kwargs) -> list:
    """Wrapper to parallelize a function.

    Data in *args and **kwargs is assumed to be list or np.ndarray. Each
    data will be divided into n_cpus chunks.

    :param n_cpus: number of cpus to execute the function
    :param shared_kwargs: dictionary which contains the argument that are
                          shared among the jobs. Those arguments are passed
                          as they are to each job.
    :param func: Callable to parallelize
    :param *args: arguments of func
    :param **kwargs: arguments passed with named keys of func

    """
    if not isinstance(shared_kwargs, dict):
        raise ValueError("`shared_args` must be a dictionary")

    n_cpus = min(n_cpus, os.cpu_count())
    if n_cpus < 1:
        raise ValueError("Number of cpus must be greater than 1")

    if n_cpus == 1:
        # Returns a list for consistency wrt the multi cpu case
        return [func(*args, **{**kwargs, **shared_kwargs})]

    valid_types = [list, np.ndarray]
    str_valid_types = " ".join([str(x) for x in valid_types])
    payloads = [{"args": [], "kwargs": {}} for _ in range(n_cpus)]

    for param_set_id, param_set in (
        ("args", enumerate(args)),
        ("kwargs", kwargs.items()),
    ):
        for key, value in param_set:
            if type(value) not in valid_types:
                raise TypeError(
                    f"Found an argument with type {str(type(value))}. Supported types are: {str_valid_types}"
                )

            if type(value) in valid_types:
                chunks = np.array_split(value, n_cpus)
                for c, p in zip(chunks, payloads):
                    if param_set_id == "args":
                        p["args"].append(c)
                    else:
                        p["kwargs"][key] = c

    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp_context) as executor:
        processes = []
        for payload in payloads:
            process_args = payload["args"]
            # Merge local process kwargs and shared kwargs dictionaries
            process_kwargs = {**payload["kwargs"], **shared_kwargs}
            processes.append(executor.submit(func, *process_args, **process_kwargs))
        results = [process.result() for process in processes]
    return results
