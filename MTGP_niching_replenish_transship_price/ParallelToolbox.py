import pickle
import sys
import os
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

        # Windows-specific fix for multiprocessing
        if sys.platform.startswith('win'):
            # Add freeze_support for Windows
            mp.freeze_support()

            # Get the executable path and ensure it exists
            python_exe = sys.executable
            if not os.path.exists(python_exe):
                # Fallback: try to find python.exe in the current environment
                python_exe = os.path.join(sys.prefix, 'python.exe')

            # Set the executable explicitly
            mp.set_executable(python_exe)

        # Set start method to 'spawn' (safer for Windows)
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Start method already set, continue
            pass

        # Number of CPU cores
        cores = mp.cpu_count()

        try:
            # Initialize the Pool with the number of cores
            with mp.Pool(processes=cores) as pool:
                # Create a list of arguments tuples where seed_value is constant
                args = [(ind, seed_value, parameters) for ind in invalid_ind]

                # Use starmap to pass multiple arguments to the function
                fitnesses = pool.starmap(evaluate, args)

            return fitnesses

        except Exception as e:
            # print(f"Error in multiprocessing: {e}")
            # print(f"Python executable: {sys.executable}")
            # print(f"Falling back to sequential processing...")

            # Fallback to sequential processing if multiprocessing fails
            fitnesses = []
            for ind in invalid_ind:
                fitness = evaluate(ind, seed_value, parameters)
                fitnesses.append(fitness)

            return fitnesses