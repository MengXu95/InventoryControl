import numpy as np
import random
import pandas as pd
import math
import MTGP_niching_replenish_transship_price.LoadIndividual as mtload
import MTGP_niching_replenish_transship_price.saveFile as mtsave
from MTGP_niching_replenish_transship_price.replenishment import *
from MTGP_niching_replenish_transship_price.transshipment import *
from MTGP_niching_replenish_transship_price.Inventory_simulator_replenish_transship_price import InvOptEnv
import os
from Utils.ScenarioDesign_replenish_transship_price import ScenarioDesign_replenish_transship_price


def main(dataset_name, run):
    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign_replenish_transship_price(dataset_name)
    parameters = scenarioDesign.get_parameter()

    num_instances = 100
    seed = 888
    seed_rotation = 88
    print('\nBegin testing GP for policy from each generation: ')

    all_gen_individuals = mtload.load_individual_from_gen(run, dataset_name)
    all_PC_diversity = mtload.load_PC_diversity(run, dataset_name)

    replenishment_rule_size = []
    transshipment_rule_size = []
    RFQ_predict_rule_size = []
    test_fitness = []
    PC_diversity = []
    final_gen_each_instance = []

    # Determine individual structure from first individual
    individual_length = len(all_gen_individuals[0])

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
            reward_total, all_cost_fit = env.run_test(individual, states=GP_states, actions=GP_actions,
                                                      rewards=GP_rewards)
            fitness += reward_total
            if idx == len(all_gen_individuals) - 1:
                final_gen_each_instance.append(reward_total)

        fitness = fitness / num_instances
        test_fitness.append(fitness)

        # Store rule sizes based on individual structure
        replenishment_rule_size.append(len(individual[0]))

        if individual_length >= 2:
            transshipment_rule_size.append(len(individual[1]))
        else:
            transshipment_rule_size.append(0)

        if individual_length >= 3:
            RFQ_predict_rule_size.append(len(individual[2]))
        else:
            RFQ_predict_rule_size.append(0)

    for row in all_PC_diversity['PCdiversity']:
        PC_diversity.append(float(row))

    # Create DataFrame based on individual structure
    if individual_length == 1:
        # Only replenishment
        df = pd.DataFrame({
            'Run': [run for x in range(len(test_fitness))],
            'Generation': [x for x in range(len(test_fitness))],
            'RepRuleSize': [x for x in replenishment_rule_size],
            'TestFitness': [x for x in test_fitness],
            'PCDiversity': [x for x in PC_diversity],
        })
    elif individual_length == 2:
        # Replenishment + Transshipment
        df = pd.DataFrame({
            'Run': [run for x in range(len(test_fitness))],
            'Generation': [x for x in range(len(test_fitness))],
            'RepRuleSize': [x for x in replenishment_rule_size],
            'TransRuleSize': [x for x in transshipment_rule_size],
            'TestFitness': [x for x in test_fitness],
            'PCDiversity': [x for x in PC_diversity],
        })
    elif individual_length == 3:
        # Replenishment + Transshipment + RFQ Prediction
        df = pd.DataFrame({
            'Run': [run for x in range(len(test_fitness))],
            'Generation': [x for x in range(len(test_fitness))],
            'RepRuleSize': [x for x in replenishment_rule_size],
            'TransRuleSize': [x for x in transshipment_rule_size],
            'RFQRuleSize': [x for x in RFQ_predict_rule_size],
            'TestFitness': [x for x in test_fitness],
            'PCDiversity': [x for x in PC_diversity],
        })
    else:
        raise ValueError(f"Unexpected individual length: {individual_length}")

    # save the test results df
    mtsave.save_TestResults_to_csv(run, dataset_name, df)

    print(f"\nTest completed for run {run}")
    print(f"Individual structure: {individual_length} tree(s)")
    print(f"Average test fitness: {np.mean(test_fitness):.4f}")
    print(f"Final generation average: {np.mean(final_gen_each_instance):.4f}")

    return df