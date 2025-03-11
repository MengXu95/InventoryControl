import pickle

from deap import base
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
        with mp.Pool(processes=cores) as pool:
            # Create a list of arguments tuples where seed_value is constant
            args = [(ind, seed_value, parameters) for ind in invalid_ind]

            # Use starmap to pass multiple arguments to the function
            fitnesses = pool.starmap(evaluate, args)

        # pickle.dumps(invalid_ind)
        # pickle.dumps(evaluate)
        # pickle.dumps(seeds)
        # fitnesses = Pool().map(evaluate, invalid_ind, seeds)
        return fitnesses





