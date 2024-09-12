import pickle
import numpy as np
import os
import csv
import torch
import pandas as pd


def save_policy(randomSeeds, dataSetName,ppo_agent, file_path=None):
    # Construct the directory and file path
    directory = f'./DRL/train/scenario_{dataSetName}/'
    if file_path == None:
        file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_policy_best_so_far.pt')


    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the individuals to the file
    with open(file_path, 'wb') as file:
        torch.save(ppo_agent.policy.state_dict(), file_path)
    return

def saveMinFitness(randomSeeds, dataSetName, min_fitness):
    # Construct the directory and file path
    directory = f'./DRL/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_min_fitness{dataSetName}.npy')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the min_fitness array to the file
    np.save(file_path, min_fitness)

    return

def saveRunningTime(randomSeeds, dataSetName, running_time):
    # Construct the directory and file path
    directory = f'./DRL/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_running_time{dataSetName}.npy')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the running_time array to the file
    np.save(file_path, running_time)

    return

def saveActionMap(randomSeeds, dataSetName, action_map):
    # Construct the directory and file path
    directory = f'./DRL/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_action_map_{dataSetName}.npy')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the running_time array to the file
    np.save(file_path, action_map)

    return

def save_TestResults_to_csv(randomSeeds, dataSetName, resultsDf):

    # Construct the directory and file path
    directory = f'./DRL/train/scenario_{dataSetName}/test/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_testResults.csv')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the DataFrame to a CSV file
    resultsDf.to_csv(file_path, index=False)

    return