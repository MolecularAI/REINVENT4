"""Run Jan H. Jensen's GA_GB algorithm"""

from multiprocessing import Pool
from typing import List

import numpy as np
from rdkit import Chem

from .GB_GA import GA
from . import crossover

crossover.average_size = 39.15
crossover.size_stdev = 3.50


def run_ga(smilies: List[str], config: dict) -> List[str]:
    """Run the GA algorithm

    :param smilies: seed SMILES
    :param config: configuration dictionary
    :returns: sampled SMILES
    """

    mols = [Chem.MolFromSmiles(smiles) for smiles in smilies]

    n_tries = config.get("batch_size", 10)
    population_size = config.get("population_size", 20)
    mating_pool_size = config.get("mating_pool_size", 20)
    generations = config.get("generations", 10)
    mutation_rate = config.get("mutation_rate", 0.05)
    n_cpus = config.get("n_cpus", 1)

    prune_population = True  # enforces unique SMILES
    max_score = 0.7  # QED

    seeds = np.random.randint(100_000, size=2 * n_tries)
    index = slice(0, n_tries) if prune_population else slice(n_tries, 2 * n_tries)

    temp_args = [
        [
            population_size,
            mols,
            generations,
            mating_pool_size,
            mutation_rate,
            max_score,
            prune_population,
        ]
        for _ in range(n_tries)
    ]

    args = []

    for x, y in zip(temp_args, seeds[index]):
        x.append(y)
        args.append(x)

    with Pool(n_cpus) as pool:
        output = pool.map(GA, args)

    smilies = [Chem.MolToSmiles(population[0], isomericSmiles=True) for _, population, _ in output]

    return smilies
