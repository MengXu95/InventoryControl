import numpy as np
import random
import pandas as pd
import math
import MTGP_niching_rental_RFQ_price.LoadIndividual as mtload
import MTGP_niching_rental_RFQ_price.saveFile as mtsave
from MTGP_niching_rental_RFQ_price.replenishment import *
from MTGP_niching_rental_RFQ_price.transshipment import *
from MTGP_niching_rental_RFQ_price.Inventory_simulator_rental_RFQ import InvOptEnv
import os
from Utils.ScenarioDesign_rental_RFQ_price import ScenarioDesign_rental_RFQ_price



def main(dataset_name, run):
    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign_rental_RFQ_price(dataset_name)
    parameters = scenarioDesign.get_parameter()

    num_instances = 100
    seed = 888
    seed_rotation = 88
    print('\nBegin testing GP for policy from each generation: ')

    all_gen_individuals = mtload.load_individual_from_gen(run, dataset_name)
    all_PC_diversity = mtload.load_PC_diversity(run, dataset_name)

    replenishment_rule_size = []
    rental_rule_size = []
    if len(all_gen_individuals[0]) == 3:
        RFQ_predict_rule_size = []
    test_fitness = []
    PC_diversity = []
    final_gen_each_instance = []
    for idx in range(len(all_gen_individuals)):
        print("Generation: " + str(idx))
        individual = all_gen_individuals.get(idx)
        fitness = 0
        GP_states = []
        GP_actions = []
        GP_rewards = []
        for ins in range(num_instances):
            env = InvOptEnv(seed, parameters)
            seed = seed + seed_rotation
            reward_total, all_cost_fit = env.run_test(individual,states=GP_states,actions=GP_actions,rewards=GP_rewards)
            fitness += reward_total
            if idx == len(all_gen_individuals)-1:
                final_gen_each_instance.append(reward_total)
        fitness = fitness/num_instances
        test_fitness.append(fitness)

        replenishment_rule_size.append(len(individual[0]))
        rental_rule_size.append(len(individual[1]))
        if len(all_gen_individuals[0]) == 3:
            RFQ_predict_rule_size.append(len(individual[2]))

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
        'RentRuleSize': [x for x in rental_rule_size],
        'TestFitness': [x for x in test_fitness],
        'PCDiversity': [x for x in PC_diversity],
    })
    if len(all_gen_individuals[0]) == 3:
        df = pd.DataFrame({
            'Run': [run for x in range(len(test_fitness))],
            'Generation': [x for x in range(len(test_fitness))],
            'RepRuleSize': [x for x in replenishment_rule_size],
            'RentRuleSize': [x for x in rental_rule_size],
            'RFQRuleSize': [x for x in RFQ_predict_rule_size],
            'TestFitness': [x for x in test_fitness],
            'PCDiversity': [x for x in PC_diversity],
            })

    # save the test results df
    mtsave.save_TestResults_to_csv(run,dataset_name,df)

