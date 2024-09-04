import numpy as np
import random
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
import CCGP_niching.LoadIndividual as mtload
import CCGP_niching.saveFile as mtsave
from CCGP_niching.replenishment import *
from CCGP_niching.transshipment import *
from CCGP_niching.Inventory_simulator import InvOptEnv
import os
from Utils.ScenarioDesign import ScenarioDesign
import threading

# For sS policy
# Implement sS action, assume LT=2
def sS_action(state, sS1, sS2):
    if state[0] + state[-2] < sS1[0]:
        order1 = sS1[1] - state[0] - state[-2]
    else:
        order1 = 0
    if state[1] + state[-1] < sS2[0]:
        order2 = sS2[1] - state[1] - state[-1]
    else:
        order2 = 0
    return order1, order2

def main(dataset_name, run):
    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign(dataset_name)
    parameters = scenarioDesign.get_parameter()

    num_instances = 50
    seed = 888
    seed_rotation = 88
    print('\nBegin testing GP for policy from each generation: ')

    all_gen_individuals = mtload.load_individual_from_gen(run, dataset_name)
    all_PC_diversity = mtload.load_PC_diversity(run, dataset_name)

    replenishment_rule_size = []
    transshipment_rule_size = []
    test_fitness = []
    PC_diversity = []
    for idx in range(len(all_gen_individuals)):
        print("Generation: " + str(idx))
        individual = all_gen_individuals.get(idx)
        fitness = 0
        for _ in range(num_instances):
            env = InvOptEnv(seed, parameters)
            seed = seed + seed_rotation
            reward_total = env.run_test(individual)
            fitness += reward_total
        fitness = fitness/num_instances
        test_fitness.append(fitness)
        replenishment_rule_size.append(len(individual[0]))
        transshipment_rule_size.append(len(individual[1]))

    for row in all_PC_diversity['PCdiversity']:
        PC_diversity.append(float(row))

    # print(replenishment_rule_size)
    # print(transshipment_rule_size)
    # print(test_fitness)
    # print(PC_diversity)

    df = pd.DataFrame({
        'Run': [run for x in range(len(test_fitness))],
        'Generation': [x for x in range(len(test_fitness))],
        'RepRuleSize': [x for x in replenishment_rule_size],
        'TraRuleSize': [x for x in transshipment_rule_size],
        'TestFitness': [x for x in test_fitness],
        'PCDiversity': [x for x in PC_diversity],
        })

    # save the test results df
    mtsave.save_TestResults_to_csv(run,dataset_name,df)

