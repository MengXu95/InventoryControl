import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
import MTGP_niching.LoadIndividual as mtload
from MTGP_niching.replenishment import *
from MTGP_niching.transshipment import *
from MTGP_niching.Inventory_simulator import InvOptEnv
import os

np.random.seed(0)
random.seed(0)

#########################################################
# Begin testing
# For GP policy
# read GP policy
num_instances = 10
num_retailer = 3
seed = 888
seed_rotation = 88
print('\nBegin testing GP for policy from each generation: ')
randomSeeds = 1
dataSetName = 'sN3h_1_5_10b2'
dict_best_individuals = mtload.load_individual_from_gen(randomSeeds, dataSetName)
# training_time = mtload.load_training_time(randomSeeds, dataSetName)
# min_fitness = mtload.load_min_fitness(randomSeeds, dataSetName)
# print('\nTraining time: ')
# print(training_time)
# print('Training min_fitness: ')
# print(min_fitness)
# print('\n')

best_invlvls_GP = [np.Infinity] # List of average inventory levels
best_fill_GP = [np.Infinity]
best_cost_GP = [np.Infinity] # List of rewards for each demand realization (DRL no transshipment scenario)
best_cost_GP_all_gens = []
final_gen_invlvls_GP = [np.Infinity] # List of average inventory levels
final_gen_fill_GP = [np.Infinity]
final_gen_cost_GP = [np.Infinity] # List of rewards for each demand realization (DRL no transshipment scenario)
for idx in range(len(dict_best_individuals)):
    print("Generation: " + str(idx))
    algo = 'gen_' + str(idx) + '_MTGP_test'
    individual = dict_best_individuals.get(idx)
    if len(individual) == 1:
        replenishment_policy = individual[0]
    else:
        replenishment_policy = individual[0]
        transshipment_policy = individual[1]
    invlvls_GP = [] # List of average inventory levels
    fill_GP = []
    cost_GP = [] # List of rewards for each demand realization (DRL no transshipment scenario)
    GP_states = []
    GP_actions = []
    GP_rewards = []
    for _ in range(num_instances):
        env = InvOptEnv(seed)
        seed = seed + seed_rotation
        reward_total = env.run_test(individual, GP_states, GP_actions, GP_rewards)
        cost_GP.append(reward_total)
        invlvls = 0
        fill = 0
        for each_state in GP_states:
            for retailer in each_state[0]:
                invlvls = invlvls + retailer[0]
                fill = fill + int(retailer[0]>=0)
        invlvls = invlvls/len(GP_states[0])
        fill = fill/len(GP_states[0])
        invlvls_GP.append(invlvls)
        fill_GP.append(fill)
    if mean(cost_GP) < mean(best_cost_GP):
        best_cost_GP = cost_GP
    if mean(invlvls_GP) < mean(best_invlvls_GP):
        best_invlvls_GP = invlvls_GP
    if mean(fill_GP) < mean(best_fill_GP):
        best_fill_GP = fill_GP
    final_gen_cost_GP = cost_GP
    final_gen_invlvls_GP = invlvls_GP
    final_gen_fill_GP = fill_GP
    print(cost_GP)
    print(mean(cost_GP))
    best_cost_GP_all_gens.append(mean(cost_GP))
    if num_retailer==2:
        df = pd.DataFrame({
                            'states': [torch.Tensor(x[0]) for x in GP_states],
                            'transshipment': [x[0] for x in GP_actions],
                            'actions1': [x[1] for x in GP_actions],
                            'actions2': [x[2] for x in GP_actions],
                            'rewards': GP_rewards,
                            })
    else: # this is for N=3
        df = pd.DataFrame({
            'states': [torch.Tensor(x[0]) for x in GP_states],
            'transshipment01': [x[0] for x in GP_actions],
            'transshipment02': [x[1] for x in GP_actions],
            'transshipment12': [x[2] for x in GP_actions],
            'actions0': [x[3] for x in GP_actions],
            'actions1': [x[4] for x in GP_actions],
            'actions2': [x[5] for x in GP_actions],
            'rewards': GP_rewards,
        })
    # Define the folder path
    folder_path = "./GP_trajectory/" + dataSetName + "/run_" + str(randomSeeds)
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
    df.to_csv(folder_path + "/GP_trajectory_gen_" + str(idx) + ".csv", index=False)

#########painting curve###########
# Plotting the values
plt.plot(best_cost_GP_all_gens, marker='o', linestyle='-', color='b')

# Adding titles and labels
plt.title('Convergence of GP')
plt.xlabel('Generation')
plt.ylabel('Cost')
# Save the plot to a PDF file
plt.savefig('convergence_of_GP.pdf')
# Display the plot
plt.show()
#########painting curve###########

print('\n-----Final comparison with final gen GP------: ')
print([mean(final_gen_cost_GP)])
print([np.std(final_gen_cost_GP)])

print('\nFinal inventory level and fill: ')
print([mean(final_gen_invlvls_GP),  mean(final_gen_fill_GP)])