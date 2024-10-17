import numpy as np
import random
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
from sSPolicy.Inventory_simulator import InvOptEnv
import os
from Utils.ScenarioDesign import ScenarioDesign
import threading


def main(dataset_name, run_seed):
    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign(dataset_name)
    parameters = scenarioDesign.get_parameter()

    if dataset_name == "teckwah_training":
        demand_level = float(30000)
    else:
        demand_level = float(parameters['demand_level'])
    interval = demand_level/10
    candidate_sS = []
    value = 0
    for i in range(1, 10):
        value = value + interval
        candidate_sS.append(value)
    candidate_sS.append(demand_level)

    all_combination = []
    num_site = len(parameters['ini_inv'])
    for i in range(len(candidate_sS)):
        for j in range(i, len(candidate_sS)):
            all_combination.append([candidate_sS[i], candidate_sS[j]]) #transportation quantity always be 0

    # random try to get the best combination of sS Policy
    pop_size = 400
    num_ins = 2
    gens = 50
    try_times = pop_size
    np.random.seed(run_seed)
    randomSeed_ngen = []
    for i in range(gens):
    # for i in range((ngen+1)*ins_each_gen): # the *ins_each_gen is added by mengxu followed the advice of Meng 2022.11.01
        seed_i = np.random.randint(2000000000)
        randomSeed_ngen.append(seed_i)
        seed_i = seed_i + 1000
        randomSeed_ngen.append(seed_i)

    print('\nBegin train sSPolicy: ')
    all_sS_policy = []
    for i in range(try_times):
        sS_policy = []
        for i in range(num_site):
            random_int_i = np.random.randint(0, len(all_combination))
            sS_policy.append(all_combination[random_int_i])
        all_sS_policy.append(sS_policy)

    best_fitness = np.inf
    best_sSPolicy = None
    for i in range(len(all_sS_policy)):
        sS_policy = all_sS_policy[i]
        fitness = 0
        for ins in range(len(randomSeed_ngen)):
            seed = randomSeed_ngen[ins]
            env = InvOptEnv(seed, parameters)
            reward_total = env.run_sSPolicy(sS_policy)
            fitness += reward_total
        fitness = fitness / len(randomSeed_ngen)
        if fitness < best_fitness:
            best_sSPolicy = sS_policy
            best_fitness = fitness


    print('Best sSPolicy: ' + str(best_sSPolicy))
    print('\nBegin test sSPolicy: ')
    num_instances = 20
    seed = 888
    seed_rotation = 88

    test_fitness_each_instance = []
    test_fitness = []
    fitness = 0
    for ins in range(num_instances):
        if dataset_name == "teckwah_training":
            parameters = scenarioDesign.get_parameter(seed=ins)
        env = InvOptEnv(seed, parameters)
        seed = seed + seed_rotation
        reward_total = env.run_sSPolicy(best_sSPolicy)
        fitness += reward_total
        test_fitness_each_instance.append(reward_total)
    fitness = fitness/num_instances
    test_fitness.append(fitness)
    print('Test results: ' + str(test_fitness))

    # save results
    sS_results = []
    sS_results.append({
        'Scenario': dataset_name,
        'Algorithm': best_sSPolicy,
        'TestFitness': fitness
    })

    # Save results to a CSV file
    sS_df = pd.DataFrame(sS_results, columns=['Scenario', 'Algorithm', 'TestFitness'])
    directory = f'./sSPolicy/train/scenario_{dataset_name}/'
    file_path = os.path.join(directory, f'{run_seed}_{dataset_name}_sSPolicy_test_results.csv')
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    sS_df.to_csv(file_path, index=False)


    # save the test results df
    if dataset_name == "teckwah_training":
        # to output the results of final gen and each instance
        df_final = pd.DataFrame({
            'Run': [run_seed for x in range(len(test_fitness_each_instance))],
            'TestFitness': [x for x in test_fitness_each_instance],
        })
        directory = f'./sSPolicy/train/scenario_{dataset_name}/'
        file_path = os.path.join(directory, f'{run_seed}_{dataset_name}_sSPolicy_test_results_each_instance.csv')
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        df_final.to_csv(file_path, index=False)

    return best_sSPolicy, fitness


    # df = pd.DataFrame({
    #     'Run': [run for x in range(len(test_fitness))],
    #     'Generation': [x for x in range(len(test_fitness))],
    #     'TestFitness': [x for x in test_fitness]
    #     })
    #
    # # save the test results df
    # print()

if __name__ == "__main__":
    workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/Results/"

    # List of small scenarios
    scenarios = ["sN2h_1_5b2", "sN2h_1_10b3", "sN2h_5_10b5", "sN2h_5_50b10",
                 "sN2h_10_50b2", "sN2h_10_100b3", "sN2h_50_100b5", "sN2h_100_100b10",
                 "sN3h_1_5_10b2", "sN3h_1_5_50b3", "sN3h_5_10_50b5", "sN3h_5_5_50b10",
                 "sN3h_10_50_50b2", "sN3h_10_50_100b3", "sN3h_50_50_50b5", "sN3h_50_50_100b10"]
    scenarios_type = 'small'
    # List of medium scenarios
    # scenarios = ["mN2h_1_5b2", "mN2h_1_10b3", "mN2h_5_10b5", "mN2h_5_50b10",
    #              "mN2h_10_50b2", "mN2h_10_100b3", "mN2h_50_100b5", "mN2h_100_100b10",
    #              "mN3h_1_5_10b2", "mN3h_1_5_50b3", "mN3h_5_10_50b5", "mN3h_5_5_50b10",
    #              "mN3h_10_50_50b2", "mN3h_10_50_100b3", "mN3h_50_50_50b5", "mN3h_50_50_100b10"]
    # scenarios_type = 'medium'
    # List of large scenarios
    # scenarios = ["lN2h_1_5b2", "lN2h_1_10b3", "lN2h_5_10b5", "lN2h_5_50b10",
    #              "lN2h_10_50b2", "lN2h_10_100b3", "lN2h_50_100b5", "lN2h_100_100b10",
    #              "lN3h_1_5_10b2", "lN3h_1_5_50b3", "lN3h_5_10_50b5", "lN3h_5_5_50b10",
    #              "lN3h_10_50_50b2", "lN3h_10_50_100b3", "lN3h_50_50_50b5", "lN3h_50_50_100b10"]
    # scenarios_type = 'large'

    # Initialize a dictionary to hold data for all algorithms and scenarios
    data = {scenario: [] for scenario in scenarios}

    for scenario in scenarios:
        best_sSPolicy, result = main(scenario)
        data[scenario].append({'Algorithm': best_sSPolicy, 'TestFitness': result})

    sS_results = []
    for scenario in scenarios:
        scenario_data = pd.DataFrame(data[scenario]).iloc[0]
        sS_results.append({
            'Scenario': scenario,
            'Algorithm': scenario_data['Algorithm'],
            'TestFitness': scenario_data['TestFitness']
        })

    # Save Mean/Std results to a CSV file
    sS_df = pd.DataFrame(sS_results, columns=['Scenario', 'Algorithm', 'TestFitness'])
    sS_df.to_csv(os.path.join(workdir, "sSPolicy_test_results_" + scenarios_type + ".csv"), index=False)