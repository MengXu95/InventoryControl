import numpy as np
import random
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
import MTGP_niching.LoadIndividual as mtload
import MTGP_niching.saveFile as mtsave
from MTGP_niching.replenishment import *
from MTGP_niching.transshipment import *
from MTGP_niching.Inventory_simulator import InvOptEnv
import os
from Utils.ScenarioDesign import ScenarioDesign



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

