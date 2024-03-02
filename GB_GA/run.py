import sys
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, rdBase

import GB_GA as ga
import crossover as co

rdBase.DisableLog("rdApp.error")


n_tries = 10  # determines number of output SMILES
population_size = 20
mating_pool_size = 20
generations = 10
mutation_rate = 0.05
co.average_size = 39.15
co.size_stdev = 3.50
max_score = 0.7  # QED
n_cpus = 4
seeds = np.random.randint(100_000, size=2 * n_tries)

file_name = sys.argv[1]

print("* RDKit version", rdBase.rdkitVersion)
print("* population_size", population_size)
print("* mating_pool_size", mating_pool_size)
print("* generations", generations)
print("* mutation_rate", mutation_rate)
print("* max_score", max_score)
print("* average_size/size_stdev", co.average_size, co.size_stdev)
print("* initial pool", file_name)
print("* number of tries", n_tries)
print("* number of CPUs", n_cpus)
print("* seeds", ",".join(map(str, seeds)))
print("* ")
print("run,score,smiles,generations,prune")

count = 0

for prune_population in [True, False]:
    index = slice(0, n_tries) if prune_population else slice(n_tries, 2 * n_tries)
    temp_args = [
        [
            population_size,
            file_name,
            generations,
            mating_pool_size,
            mutation_rate,
            max_score,
            prune_population,
        ]
        for i in range(n_tries)
    ]
    args = []

    for x, y in zip(temp_args, seeds[index]):
        x.append(y)
        args.append(x)

    with Pool(n_cpus) as pool:
        output = pool.map(ga.GA, args)

    for i in range(n_tries):
        scores, population, generation = output[i]
        smiles = Chem.MolToSmiles(population[0], isomericSmiles=True)
        print(f"{i},{scores[0]:.2f},{smiles},{generation},{prune_population}")
