import pickle

from deap import base
from multiprocessing import cpu_count, Pool

import multiprocessing as mp


##thanks TPOT
## https://github.com/EpistasisLab/tpot/pull/100/files
class ParallelToolbox(base.Toolbox):
    """Runs the TPOT genetic algorithm over multiple cores."""

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    # created by mengxu 2022.11.28 for multiple processing
    def multiProcess(self, evaluate, invalid_ind, seed_value, parameters):
        # Number of CPU cores
        cores = mp.cpu_count()
        # Initialize the Pool with the number of cores
        fitnesses = []
        with mp.Pool(processes=cores) as pool:
            # Assume best_ind1 and best_ind2 are the best individuals from the previous generation
            subpop1 = invalid_ind[0]
            subpop2 = invalid_ind[1]
            best_ind1 = subpop1[0] # the subpop has already been sorted
            best_ind2 = subpop2[0] # the subpop has already been sorted

            combined_ind_subpop1 = []
            for ind1 in subpop1:
                # Evaluate ind1 with best_ind2 from subpop2
                combined_solution = [ind1, best_ind2]
                combined_ind_subpop1.append(combined_solution)

            # Create a list of arguments tuples where seed_value is constant
            args = [(ind, seed_value, parameters) for ind in combined_ind_subpop1]

            # Use starmap to pass multiple arguments to the function
            fitnesses_subpop1 = pool.starmap(evaluate, args)
            fitnesses.append(fitnesses_subpop1)

        with mp.Pool(processes=cores) as pool:
            # Assume best_ind1 and best_ind2 are the best individuals from the previous generation
            subpop1 = invalid_ind[0]
            subpop2 = invalid_ind[1]
            best_ind1 = subpop1[0]  # the subpop has already been sorted
            best_ind2 = subpop2[0]  # the subpop has already been sorted

            combined_ind_subpop2 = []
            for ind2 in subpop2:
                # Evaluate ind1 with best_ind2 from subpop2
                combined_solution = [best_ind1, ind2]
                combined_ind_subpop2.append(combined_solution)

            # Create a list of arguments tuples where seed_value is constant
            args = [(ind, seed_value, parameters) for ind in combined_ind_subpop2]

            # Use starmap to pass multiple arguments to the function
            fitnesses_subpop2 = pool.starmap(evaluate, args)
            fitnesses.append(fitnesses_subpop2)

        return fitnesses

    def select_best(self, subpopulation):
        """
        Selects the best individual from a subpopulation based on fitness.

        Parameters:
        subpopulation (list): A list of individuals, each having a fitness attribute.

        Returns:
        best_individual: The individual with the best (lowest) fitness in the subpopulation.
        """
        best_individual = min(subpopulation, key=lambda ind: ind.fitness.values[0])
        return best_individual

