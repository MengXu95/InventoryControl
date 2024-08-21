import simpy
from deap import base
from deap import creator
from deap import gp
import MTGP_niching.multi_tree as mt
from MTGP_niching import ea_simple_elitism
from MTGP_niching.ParallelToolbox import ParallelToolbox
from MTGP_niching.selection import *
import sys
from MTGP_niching import saveFile
import time
import random
from MTGP_niching.Inventory_simulator import *

import numpy as np



def connectedness(cluster):
    print(cluster)


def init_toolbox(toolbox, pset):
    REP.init_toolbox(toolbox, pset, N_TREES)
    toolbox.register("select", selElitistAndTournament, tournsize=7, elitism=ELITISM)

def init_toolbox_two_pset(toolbox, pset1, pset2):
    REP.init_toolbox_two_pset(toolbox, pset1, pset2, N_TREES)
    toolbox.register("select", selElitistAndTournament, tournsize=7, elitism=ELITISM)


def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats

def evaluate(individual,seed):
    # add by mengxu 2022.10.13 to add the training instances ===============================================
    # create the environment instance for simulation
    # Generate forecasts and demand
    # seed = rd['seed']
    env = InvOptEnv(seed)
    fitness = env.run(individual)

    for i in range(ins_each_gen-1):
        env.reset()
        fitness_i = env.run(individual)

        fitness = fitness + fitness_i

    # spf.job_creator.final_output() #for check
    fitness = fitness/ins_each_gen
    scores = [fitness]
    return scores


def eval_wrapper(*args, **kwargs):
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], seed = rd['seed'])
    return evaluate(*args, **kwargs)
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], data=rd['data'], labels=rd['labels'])


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata


def GPFC_main(dataset_name, seed):

    rd['seed'] = seed
    rd['dataset_name'] = dataset_name
    num_features = 0 # the initial number of terminals is 0, then I will add more terminals into the pset
    if DIFF_PSET:
        pset1 = gp.PrimitiveSet("MAIN1", num_features, prefix="f")
        pset1.context["array"] = np.array
        REP.init_primitives_replenishment(pset1)
        pset2 = gp.PrimitiveSet("MAIN2", num_features, prefix="f")
        pset2.context["array"] = np.array
        REP.init_primitives_transshipment(pset2)
        weights = (-1.,)
        creator.create("FitnessMin", base.Fitness, weights=weights)
        # set up toolbox
        toolbox = ParallelToolbox()  # base.Toolbox()
        init_toolbox_two_pset(toolbox, pset1, pset2)
        toolbox.register("evaluate", eval_wrapper)
    else:
        pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
        pset.context["array"] = np.array
        REP.init_primitives(pset)
        weights = (-1.,)
        creator.create("FitnessMin", base.Fitness, weights=weights)
        # set up toolbox
        toolbox = ParallelToolbox()  # base.Toolbox()
        init_toolbox(toolbox, pset)
        toolbox.register("evaluate", eval_wrapper)

    rd['toolbox'] = toolbox
    pop = toolbox.population(n=POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    # seedRotate = False # added by mengxu 2022.10.13
    pop, logbook, min_fitness, best_ind_all_gen = ea_simple_elitism.eaSimple(pop, toolbox, CXPB, MUTPB, REPPB, ELITISM, NGEN, seedRotate, rd, stats, halloffame=hof, verbose=True, seed =seed, dataset_name=dataset_name)
    best = hof[0]
    return min_fitness,best, best_ind_all_gen


POP_SIZE =100
NGEN = 10
CXPB = 0.8
MUTPB = 0.15
REPPB = 0.05
ELITISM = 10
MAX_HEIGHT = 8
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
N_TREES = 2
rd = {}
DIFF_PSET = True
seedRotate = True # added by mengxu 2022.10.13

# create the shop floor instance
ins_each_gen = 2 # added by mengxu followed the advice of Meng 2022.11.01
def main(dataset_name, seed):
# if __name__ == "__main__":
#     dataset_name = str(sys.argv[1])
#     seed = int(sys.argv[2])
    random.seed(int(seed))
    np.random.seed(int(seed))
    saveFile.clear_individual_each_gen_to_txt(seed, dataset_name)
    start = time.time()
    min_fitness,p_one,best_ind_all_gen= GPFC_main(dataset_name,seed)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_meng(seed, dataset_name, best_ind_all_gen)
    saveFile.saveMinFitness(seed, dataset_name, min_fitness)
    saveFile.saveRunningTime(seed, dataset_name, running_time)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')

