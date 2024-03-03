"""
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
"""

import random

from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import numpy as np

from . import crossover as co
from . import mutate as mu

CALCULATOR = MolecularDescriptorCalculator(["qed"]).CalcDescriptors


def make_initial_population(population_size, mol_list):
    population = []

    for i in range(population_size):
        population.append(random.choice(mol_list))

    return population


def calculate_normalized_fitness(scores):
    sum_scores = sum(scores)
    normalized_fitness = [score / sum_scores for score in scores]

    return normalized_fitness


def make_mating_pool(population, fitness, mating_pool_size):
    mating_pool = []

    for i in range(mating_pool_size):
        mating_pool.append(np.random.choice(population, p=fitness))

    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate):
    new_population = []

    while len(new_population) < population_size:
        parent_A = random.choice(mating_pool)
        parent_B = random.choice(mating_pool)
        new_child = co.crossover(parent_A, parent_B)

        if new_child != None:
            mutated_child = mu.mutate(new_child, mutation_rate)

            if mutated_child != None:
                new_population.append(mutated_child)

    return new_population


def sanitize(population, scores, population_size, prune_population):
    if prune_population:
        smiles_list = set()
        population_tuples = []

        for score, mol in zip(scores, population):
            smiles = Chem.MolToSmiles(mol)
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

            if smiles not in smiles_list:
                smiles_list.add(smiles)
                population_tuples.append((score, mol))
    else:
        population_tuples = list(zip(scores, population))

    population_sort = sorted(population_tuples, key=lambda x: x[0], reverse=True)
    population_tuples = population_sort[:population_size]

    new_population = [t[1] for t in population_tuples]
    new_scores = [t[0] for t in population_tuples]

    return new_population, new_scores


def calculate_scores(population):
    scores = [CALCULATOR(gene)[0] for gene in population]

    return scores


def GA(args):
    (
        population_size,
        mol_list,
        generations,
        mating_pool_size,
        mutation_rate,
        max_score,
        prune_population,
        seed,
    ) = args

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)

    population = make_initial_population(population_size, mol_list)
    scores = calculate_scores(population)

    # reorder so best score comes first
    population, scores = sanitize(population, scores, population_size, False)

    fitness = calculate_normalized_fitness(scores)

    for generation in range(generations):
        mating_pool = make_mating_pool(population, fitness, mating_pool_size)
        new_population = reproduce(mating_pool, population_size, mutation_rate)
        new_scores = calculate_scores(new_population)
        population, scores = sanitize(
            population + new_population, scores + new_scores, population_size, prune_population
        )
        fitness = calculate_normalized_fitness(scores)

        if scores[0] >= max_score:
            break

    return scores, population, generation + 1
