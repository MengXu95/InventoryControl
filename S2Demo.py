import numpy as np
import random
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
import MTGP_niching_rental.LoadIndividual as mtload
import MTGP_niching_rental.saveFile as mtsave
from MTGP_niching_rental.replenishment import *
from MTGP_niching_rental.transshipment import *
from MTGP_niching_rental.Inventory_simulator_rental import InvOptEnv
import os
from Utils.ScenarioDesign_rental import ScenarioDesign_rental


def main(dataset_name, run):
    # Get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign_rental(dataset_name)
    parameters = scenarioDesign.get_parameter_S2Demo(epi_len=10)

    num_instances = 1
    seed = 8888
    seed_rotation = 88
    print('\nBegin testing GP for policy from each generation: ')

    # Load individuals and diversity data
    all_gen_individuals = mtload.load_individual_from_gen(run, dataset_name)
    all_PC_diversity = mtload.load_PC_diversity(run, dataset_name)

    test_fitness = []

    # Get the last individual from the loaded generations
    individual = all_gen_individuals.get(len(all_gen_individuals) - 1)

    # Initialize result storage
    inventory_level_0, replenishment_decision_0 = [], []
    inventory_level_1, replenishment_decision_1 = [], []
    rental_decision, total_cost = [], []
    GP_states, GP_actions, GP_rewards = [], [], []

    # Create the environment and test the individual
    env = InvOptEnv(seed, parameters)
    seed += seed_rotation  # Update seed for potential further runs
    fitness = env.run_test(individual, states=GP_states, actions=GP_actions, rewards=GP_rewards)

    test_fitness.append(fitness)

    # Store results if there are two retailers
    if parameters['num_retailer'] == 2:
        inventory_level_0 = [x[0][0][0] for x in GP_states]
        replenishment_decision_0 = [x[1] for x in GP_actions]
        inventory_level_1 = [x[0][1][0] for x in GP_states]
        replenishment_decision_1 = [x[2] for x in GP_actions]
        rental_decision = [x[3] for x in GP_actions]
        total_cost = [-x for x in GP_rewards]

    # Create DataFrame and save results
    df = pd.DataFrame({
        'inventory_level_0': inventory_level_0,
        'replenishment_decision_0': replenishment_decision_0,
        'inventory_level_1': inventory_level_1,
        'replenishment_decision_1': replenishment_decision_1,
        'rental_decision': rental_decision,
        'total_cost': total_cost,
    })

    mtsave.save_S2Demo_to_csv(run, dataset_name, df)
    print("Complete!")
