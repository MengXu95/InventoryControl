import numpy as np
import random
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
from DRL.PPO_policy import PPO
import os
from Utils.ScenarioDesign import ScenarioDesign
import threading
import torch
from DRL.Inventory_simulator import *
from DRL import saveFile
import itertools



def main(dataset_name, run):
    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign(dataset_name)
    parameters = scenarioDesign.get_parameter()

    num_instances = 50
    seed = 888
    seed_rotation = 88
    print('\nBegin testing DRL for best policy: ')

    ################################## set device ##################################
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if False:  # (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    # load action map based on the scenario
    directory = f'./DRL/train/scenario_{dataset_name}/'
    file_path = os.path.join(directory, f'{run}_action_map_{dataset_name}.npy')
    action_map = np.load(file_path)

    ################ PPO hyperparameters ################
    K_epochs = 20  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.00005  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    state_dim = 8 * parameters['num_retailer'] + 2
    action_dim = len(action_map)
    #####################################################

    # Load PPO policy
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
    directory = f'./DRL/train/scenario_{dataset_name}/'
    file_path = os.path.join(directory, f'{run}_{dataset_name}_policy_best_so_far.pt')
    # file_path = os.path.join(directory, f'{run}_{dataset_name}_policy_final.pt')
    ppo_agent.policy.load_state_dict(torch.load(file_path))
    # ppo_agent.policy.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

    replenishment_rule_size = []
    transshipment_rule_size = []
    test_fitness = []
    mean_invlvls_DRL = []
    mean_fill_DRL = []
    PC_diversity = []

    fitness = 0
    invlvls_DRL = []
    fill_DRL = []
    DRL_states = []
    DRL_actions = []
    DRL_rewards = []
    for _ in range(num_instances):
        env = InvOptEnv(seed, parameters)
        seed = seed + seed_rotation
        reward_total = env.run_test(ppo_agent,action_map,states=DRL_states,actions=DRL_actions,rewards=DRL_rewards)
        fitness += reward_total
    fitness = fitness/num_instances
    test_fitness.append(fitness)
    if parameters['num_retailer'] == 2:
        invlvls_DRL.append(mean([x[0] + x[8] for x in DRL_states]))
        fill_DRL.append(mean([int(x[0] >= 0) + int(x[8] >= 0) for x in DRL_states]))
    elif parameters['num_retailer'] == 3:
        invlvls_DRL.append(mean([x[0] + x[8] + x[16] for x in DRL_states]))
        fill_DRL.append(mean([int(x[0] >= 0) + int(x[8] >= 0) + int(x[16] >= 0) for x in DRL_states]))
    mean_invlvls_DRL.append(mean(invlvls_DRL))
    mean_fill_DRL.append(mean(fill_DRL))

    replenishment_rule_size.append(256)
    transshipment_rule_size.append(256)
    PC_diversity.append(0)



    df = pd.DataFrame({
        'Run': [run for x in range(len(test_fitness))],
        'Generation': [x for x in range(len(test_fitness))],
        'RepRuleSize': [x for x in replenishment_rule_size],
        'TraRuleSize': [x for x in transshipment_rule_size],
        'TestFitness': [x for x in test_fitness],
        'InveLevel': [x for x in mean_invlvls_DRL],
        'Fill': [x for x in mean_fill_DRL],
        'PCDiversity': [x for x in PC_diversity],
        })

    # save the test results df
    saveFile.save_TestResults_to_csv(run,dataset_name,df)
    print('Test done!')

