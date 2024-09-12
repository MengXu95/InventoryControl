import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import pandas as pd
import pytz
import math
from datetime import datetime
from torch.distributions import Categorical
import random

from DRL import saveFile
from Utils.ScenarioDesign import ScenarioDesign
from DRL.Inventory_simulator import *
import time
import os
from DRL.PPO_policy import PPO



################################### Training ###################################
def train(dataset_name,seed):
    print("============================================================================================")
    # PPO Implementation using PyTorch
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

    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign(dataset_name)
    parameters = scenarioDesign.get_parameter()

    # design action map based on the scenario
    demand_level = float(parameters['demand_level'])
    interval = demand_level / 10
    candidate_replenishment = []
    candidate_transshipment = []
    value = 0
    for i in range(1, 10):
        value = value + interval
        candidate_replenishment.append(value)
        candidate_transshipment.append(value)
        candidate_transshipment.append(-value)
    candidate_replenishment.append(demand_level)
    candidate_transshipment.append(demand_level)
    candidate_transshipment.append(-demand_level)

    action_lists = []
    for i in range(parameters['num_retailer']):
        for j in range(i+1,parameters['num_retailer']):
            action_lists.append(candidate_transshipment)
    for i in range(parameters['num_retailer']):
        action_lists.append(candidate_replenishment)
    action_map_candidate = [x for x in itertools.product(*action_lists)]

    # Shuffle the list
    random.shuffle(action_map_candidate)
    # Get a subset, by mengxu to avoid using too much memary
    num = np.min([30, len(action_map_candidate)])
    action_map = action_map_candidate[:num]
    saveFile.saveActionMap(seed, dataset_name, action_map) # to use the same actions when test


    epi_len = parameters['epi_len']
    max_ep_len = epi_len # max timesteps in one episode
    max_training_timesteps = int(max_ep_len * 20000)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    plot_freq = max_training_timesteps * 100      # plots, always not plot
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len       # update policy every n timesteps, was /2
    K_epochs = 20               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.00005       # learning rate for actor network
    lr_critic = 0.0001       # learning rate for critic network

    random_seed = seed        # set random seed if required (0 = no random seed)
    #####################################################

    L = parameters['L']
    LT = parameters['LT']
    # state_dim = 2 * (1 + L + LT - 1)
    state_dim = 8 * parameters['num_retailer'] + 2
    action_dim = len(action_map)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    # track total training time
    start_time = datetime.now(pytz.timezone('Asia/Singapore')).replace(microsecond=0)
    print("Started training at (SGT) : ", start_time)

    print("============================================================================================")


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 1
    cost_list = []
    best_so_far = 10**10
    
    # training loop
    if True:
        # Generate forecasts and demand
        env = InvOptEnv(seed, parameters)
        while time_step <= max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for _ in range(1, max_ep_len+1):

                # select action with policy

                action = ppo_agent.select_action(state)
                state, reward, done = env.step(action_map, action)
            
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # printing average reward
                if time_step % print_freq == 0:
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
                    cost_list.append(-print_avg_reward)
                    # print("Episode : {} \t\t Timestep : {} \t\t Average Cost : {}".format(i_episode, time_step, -print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0
                    if -print_avg_reward < best_so_far:
                        print("Best so far")
                        best_so_far = -print_avg_reward
                        saveFile.save_policy(seed, dataset_name, ppo_agent)
                        # torch.save(ppo_agent.policy.state_dict(), 'policy_best_so_far.pt')

                # plot graphs and write to csv
                if time_step % plot_freq == 0:
                    plotdone = False
                    while not plotdone:
                        try:
                            plt.figure(figsize=(15, 10))
                            plt.subplot(131) # States
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(x[0]) for x in ppo_agent.buffer.states])
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(x[1]) for x in ppo_agent.buffer.states])
                            plt.subplot(132) # Actions
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(action_map[x][1]) for x in ppo_agent.buffer.actions])
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(action_map[x][2]) for x in ppo_agent.buffer.actions])
                            plt.subplot(133) # Rewards
                            plt.plot(np.arange(1,max_ep_len+1,1),[-int(x) for x in ppo_agent.buffer.rewards])
                            directory = f'./DRL/train/scenario_{dataset_name}/'
                            file_path = os.path.join(directory, f'{seed}_{dataset_name}_plot_reward.png')
                            plt.savefig(file_path)
                            plt.close()

                            # Plot graph of costs by episode
                            plt.plot(10*np.arange(1,len(cost_list)+1,1),cost_list)
                            directory = f'./DRL/train/scenario_{dataset_name}/'
                            file_path = os.path.join(directory, f'{seed}_{dataset_name}_plot_cost.png')
                            plt.savefig(file_path)
                            plt.close()
                            #print([a.cpu() for a in ppo_agent.buffer.states])
                            #print([int(action_map[x][0]) for x in ppo_agent.buffer.actions])
                            # write to csv
                            df = pd.DataFrame({
                            'states': [a.cpu() for a in ppo_agent.buffer.states],
                            'transship': [int(action_map[x][0]) for x in ppo_agent.buffer.actions],
                            'actions1': [int(action_map[x][1]) for x in ppo_agent.buffer.actions],
                            'actions2': [int(action_map[x][2]) for x in ppo_agent.buffer.actions],
                            'logprobs': [a.cpu() for a in ppo_agent.buffer.logprobs],
                            'rewards': ppo_agent.buffer.rewards,
                            'is_terminals': ppo_agent.buffer.is_terminals
                            })
                            directory = f'./DRL/train/scenario_{dataset_name}/'
                            file_path = os.path.join(directory, f'{seed}_{dataset_name}_trajectory.csv')
                            df.to_csv(file_path, index=False)
                            plotdone = True

                            file_path = os.path.join(directory, f'{seed}_{dataset_name}_policy.pt')
                            saveFile.save_policy(seed,dataset_name,ppo_agent,file_path)
                            # torch.save(ppo_agent.policy.state_dict(), 'policy_non_trans.pt')

                        except ValueError:
                            continue
                # update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()
                # break; if the episode is over
                if done:
                    directory = f'./DRL/train/scenario_{dataset_name}/'
                    file_path = os.path.join(directory, f'{seed}_{dataset_name}_policy_final.pt')
                    saveFile.save_policy(seed, dataset_name, ppo_agent, file_path)
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

# if __name__ == '__main__':
#     train()

def main(dataset_name, seed):

    random.seed(int(seed))
    np.random.seed(int(seed))
    start = time.time()
    train(dataset_name,seed)
    end = time.time()
    running_time = end - start
    saveFile.saveRunningTime(seed, dataset_name, running_time)
    print("Training time: " + str(running_time))
    print('Training end!')