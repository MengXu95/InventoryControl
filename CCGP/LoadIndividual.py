import pickle
import numpy as np
import sys


def load_individual_from_gen(randomSeeds, dataSetName): # save individual as txt by mengxu
    # with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(
    #         randomSeeds) + '_meng_individual_' + dataSetName + '.pkl',
    #           "rb") as fileName_individual:
    with open(sys.path[0] + '/CCGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl',
            "rb") as fileName_individual:
        dict = pickle.load(fileName_individual)

    # print(dict.items())
    return dict

def load_training_time(randomSeeds, dataSetName): # save individual as txt by mengxu
    folder = './CCGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_running_time' + dataSetName + '.npy'
    training_time = np.load(folder)

    return training_time

def load_min_fitness(randomSeeds, dataSetName): # save individual as txt by mengxu
    folder = './CCGP/train/scenario_' + str(dataSetName) + '/' +  str(
        randomSeeds) + '_min_fitness' + dataSetName + '.npy'
    min_fitness = np.load(folder)

    return min_fitness
