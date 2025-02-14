import simpy
from deap import base
from deap import creator
from deap import gp
import MTGP_niching_rental.multi_tree as mt
from MTGP_niching_rental import ea_simple_elitism
from MTGP_niching_rental.ParallelToolbox import ParallelToolbox
from MTGP_niching_rental.selection import *
import sys
from MTGP_niching_rental import saveFile
import time
import random
from MTGP_niching_rental.Inventory_simulator_rental import *

import numpy as np



def connectedness(cluster):
    print(cluster)


def init_toolbox(toolbox, pset):
    REP.init_toolbox(toolbox, pset, N_TREES)
    toolbox.register("select", selElitistAndTournament, tournsize=TOURNSIZE, elitism=ELITISM)

def init_toolbox_two_pset(toolbox, pset1, pset2):
    REP.init_toolbox_two_pset(toolbox, pset1, pset2, N_TREES)
    toolbox.register("select", selElitistAndTournament, tournsize=TOURNSIZE, elitism=ELITISM)


def init_stats():
    # fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    fitness_stats = tools.Statistics(lambda ind: np.sum(ind.fitness.values))  # Compute mean first
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats

def evaluate(individual,seed,parameters):
    # add by mengxu 2022.10.13 to add the training instances ===============================================
    # create the environment instance for simulation
    # Generate forecasts and demand
    # seed = rd['seed']
    env = InvOptEnv(seed,parameters)
    fitness, all_cost = env.run(individual)
    all_cost = np.array(all_cost)

    for i in range(ins_each_gen-1):
        env.reset()
        fitness_i, all_cost_i = env.run(individual)
        all_cost_i = np.array(all_cost_i)

        fitness = fitness + fitness_i
        all_cost = all_cost + all_cost_i

    # spf.job_creator.final_output() #for check
    fitness = fitness/ins_each_gen
    all_cost = np.array(all_cost)
    all_cost = all_cost / ins_each_gen
    # scores = [fitness]
    scores = all_cost
    return scores


def eval_wrapper(*args, **kwargs):
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], seed = rd['seed'])
    return evaluate(*args, **kwargs)
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], data=rd['data'], labels=rd['labels'])


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata


def GPFC_main(dataset_name, seed, randomSeed_ngen):

    rd['seed'] = seed
    rd['dataset_name'] = dataset_name
    num_features = 0 # the initial number of terminals is 0, then I will add more terminals into the pset
    if DIFF_PSET and N_TREES == 2:
        pset1 = gp.PrimitiveSet("MAIN1", num_features, prefix="f")
        pset1.context["array"] = np.array
        REP.init_primitives_replenishment(pset1)
        pset2 = gp.PrimitiveSet("MAIN2", num_features, prefix="f")
        pset2.context["array"] = np.array
        REP.init_primitives_rental(pset2)
        # weights = (-1.,)
        weights = (-1.,-1.,-1.,-1.,)
        creator.create("FitnessMin", base.Fitness, weights=weights)
        # set up toolbox
        toolbox = ParallelToolbox()  # base.Toolbox()
        init_toolbox_two_pset(toolbox, pset1, pset2)
        toolbox.register("evaluate", eval_wrapper)
    else:
        pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
        pset.context["array"] = np.array
        REP.init_primitives_replenishment(pset)
        # weights = (-1.,)
        weights = (-1.,-1.,-1.,-1.,)
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
    pop, logbook, min_fitness, best_ind_all_gen, min_all_cost = ea_simple_elitism.eaSimple(randomSeed_ngen, pop, toolbox, CXPB, MUTPB, REPPB, ELITISM, NGEN, seedRotate, USE_Niching, rd, stats, halloffame=hof, verbose=True, seed =seed, dataset_name=dataset_name)
    best = hof[0]
    return min_fitness,best, best_ind_all_gen, min_all_cost


POP_SIZE = 200
NGEN = 50
CXPB = 0.8
MUTPB = 0.15
REPPB = 0.05
ELITISM = 2
TOURNSIZE = 5
MAX_HEIGHT = 8
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
#still only two trees, but one for replenishment, one for rental, no transshipment
N_TREES = 2
rd = {}
DIFF_PSET = True
seedRotate = False # added by mengxu 2022.10.13
USE_Niching = False

# create the shop floor instance
ins_each_gen = 1 # added by mengxu followed the advice of Meng 2022.11.01
def main(dataset_name, seed):
# if __name__ == "__main__":
#     dataset_name = str(sys.argv[1])
#     seed = int(sys.argv[2])
    random.seed(int(seed))
    np.random.seed(int(seed))
    randomSeed_ngen = []
    for i in range((NGEN + 1)):
    # for i in range((ngen+1)*ins_each_gen): # the *ins_each_gen is added by mengxu followed the advice of Meng 2022.11.01
        randomSeed_ngen.append(np.random.randint(2000000000))
    saveFile.clear_individual_each_gen_to_txt(seed, dataset_name)
    start = time.time()
    min_fitness,p_one,best_ind_all_gen, min_all_cost= GPFC_main(dataset_name,seed,randomSeed_ngen)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_meng(seed, dataset_name, best_ind_all_gen)
    saveFile.saveMinFitness(seed, dataset_name, min_fitness)
    saveFile.saveMinAllCost(seed, dataset_name, min_all_cost)
    saveFile.saveRunningTime(seed, dataset_name, running_time)
    print("min_all_cost: ")
    print(min_all_cost)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')

