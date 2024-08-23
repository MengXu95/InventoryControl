import math
import random

import numpy as np

from deap import tools


def sel_random(individuals, buckets, k):
    chosen_inds = []
    chosen_buckets = []
    for i in range(k):
        r = random.randrange(len(individuals))
        chosen_inds.append(individuals[r])
        chosen_buckets.append(buckets[r])

    return chosen_inds, chosen_buckets


def sel_least_complex(individuals, complexity_func):
    if len(individuals) == 1:
        return individuals[0]
    else:
        lowest_complexity = math.inf
        for ind in individuals:
            complexity = complexity_func(ind)
            if complexity < lowest_complexity:
                lowest_complexity = complexity
                least_complex = ind
        return least_complex


# def selElitistAndTournament(individuals, k, tournsize, elitism):
#     return tools.selBest(individuals, elitism) + tools.selTournament(individuals, k - elitism, tournsize)


def selRandom(individuals, k): # add by mengxu
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]

def selTournament(individuals, k, tournsize): # add by mengxu
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        aspirants_fit = [np.sum(ind.fitness.values) for ind in aspirants]
        best_index = np.argmin(aspirants_fit)
        chosen.append(aspirants[best_index])
    return chosen

def selElitistAndTournament(individuals, k, tournsize, elitism):
    return selTournament(individuals, k, tournsize) # modified by mengxu
    # return tools.selBest(individuals, elitism) + tools.selTournament(individuals, k - elitism, tournsize) #original