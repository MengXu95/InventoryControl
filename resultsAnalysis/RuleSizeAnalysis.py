import sys
from tabulate import tabulate
import pandas as pd
import numpy as np
import MTGP_niching.LoadIndividual as mtload
from wilcoxonTest.wilcoxonTest import doWilcoxonTest

def mean_of_best_rule_size(dataset_name):

    runs = 30
    best_MTGP_rule_index = 50
    all_best_sequencing_size = []
    all_best_routing_size = []
    all_best_sequencing_size_RL = []
    all_best_routing_size_RL = []

    for i in range(runs):
        # MTGP rule test, test the best rule obtained from all the generations
        dict_best_MTGP_individuals = mtload.load_individual_from_gen(i, dataset_name)
        individual = dict_best_MTGP_individuals.get(best_MTGP_rule_index)
        sequencing_rule = individual[0]
        routing_rule = individual[1]
        all_best_sequencing_size.append(len(sequencing_rule))
        all_best_routing_size.append(len(routing_rule))
        all_best_sequencing_size_RL.append(180)
        all_best_routing_size_RL.append(56)


    routing_mean = np.mean(all_best_routing_size)
    routing_std = np.std(all_best_routing_size)
    print('Routing: ' + str(routing_mean) + "(" + str(routing_std) + ")")
    print('Compare GP with RL:')
    res = doWilcoxonTest(all_best_routing_size, all_best_routing_size_RL, 0.05)
    if res == 0:
        print('GP = RL')
    elif res == 1:
        print('GP is significantly better than RL')
    elif res == 2:
        print('RL is significantly better than GP')
    else:
        print('There are something wrong here!')

    sequencing_mean = np.mean(all_best_sequencing_size)
    sequencing_std = np.std(all_best_sequencing_size)
    print('\nSequencing: ' + str(sequencing_mean) + "(" + str(sequencing_std) + ")")
    print('Compare GP with RL:')
    res = doWilcoxonTest(all_best_sequencing_size, all_best_sequencing_size_RL, 0.05)
    if res == 0:
        print('GP = RL')
    elif res == 1:
        print('GP is significantly better than RL')
    elif res == 2:
        print('RL is significantly better than GP')
    else:
        print('There are something wrong here!')

def mean_of_best_rule_size_single_run(dataset_name, i):

    all_best_replenishment_size = []
    all_best_transshipment_size = []

    # MTGP rule test, test the best rule obtained from all the generations
    dict_best_MTGP_individuals = mtload.load_individual_from_gen(i, dataset_name)
    best_MTGP_rule_index = len(dict_best_MTGP_individuals)-1
    individual = dict_best_MTGP_individuals.get(best_MTGP_rule_index)
    replenishment_rule = individual[0]
    transshipment_rule = individual[1]
    all_best_replenishment_size.append(len(replenishment_rule))
    all_best_transshipment_size.append(len(transshipment_rule))

    replenishment_mean = np.mean(all_best_replenishment_size)
    replenishment_std = np.std(all_best_replenishment_size)
    print('\nReplenishment: ' + str(replenishment_mean) + "(" + str(replenishment_std) + ")")

    transshipment_mean = np.mean(all_best_transshipment_size)
    transshipment_std = np.std(all_best_transshipment_size)
    print('Transshipment: ' + str(transshipment_mean) + "(" + str(transshipment_std) + ")")


if __name__ == "__main__":
    all_dataset_name = ['sN3h_1_5_10b2']
    run = 1
    sys.path[0] = "C:/Users/I3Nexus/PycharmProjects/InventoryControl/"
    for i in range(len(all_dataset_name)):
        dataset_name = all_dataset_name[i]
        print('\nResult on dataset: ' + all_dataset_name[i])
        mean_of_best_rule_size_single_run(dataset_name, run)
        # mean_of_best_rule_size(dataset_name)

