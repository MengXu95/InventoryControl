import pickle
import numpy as np
import sys
import pandas as pd


def load_individual_from_gen(randomSeeds, dataSetName): # save individual as txt by mengxu
    # with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(
    #         randomSeeds) + '_meng_individual_' + dataSetName + '.pkl',
    #           "rb") as fileName_individual:
    if dataSetName == "teckwah_test":
        dataSetName = "teckwah_training"
    with open(sys.path[0] + '/MTGP_niching/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl',
            "rb") as fileName_individual:
        dict = pickle.load(fileName_individual)

    # print(dict.items())
    return dict

def load_individual_scenario_sensitivity_analysis_from_gen(randomSeeds, scenario_dataSetName, dataSetName): # save individual as txt by mengxu
    # with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(
    #         randomSeeds) + '_meng_individual_' + dataSetName + '.pkl',
    #           "rb") as fileName_individual:
    if dataSetName == "teckwah_test":
        dataSetName = "teckwah_training"
    with open(sys.path[0] + '/MTGP_niching/train/scenario_' + str(scenario_dataSetName) + '/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl',
            "rb") as fileName_individual:
        dict = pickle.load(fileName_individual)

    # print(dict.items())
    return dict

def load_training_time(randomSeeds, dataSetName): # save individual as txt by mengxu
    folder = './MTGP_niching/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_running_time' + dataSetName + '.npy'
    training_time = np.load(folder)

    return training_time

def load_min_fitness(randomSeeds, dataSetName): # save individual as txt by mengxu
    folder = './MTGP_niching/train/scenario_' + str(dataSetName) + '/' +  str(
        randomSeeds) + '_min_fitness' + dataSetName + '.npy'
    min_fitness = np.load(folder)

    return min_fitness

def load_PC_diversity(randomSeeds, dataSetName): # save individual as txt by mengxu
    if dataSetName == "teckwah_test":
        dataSetName = "teckwah_training"
    folder = './MTGP_niching/train/scenario_' + str(dataSetName) + '/' +  str(
        randomSeeds) + '_' + dataSetName + '_PCdiversity.csv'
    csv_reader = pd.read_csv(folder)

    return csv_reader