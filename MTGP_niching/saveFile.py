import pickle
import numpy as np
import os
import csv
import pandas as pd


def save_individual(randomSeeds, dataSetName,individuals):
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}.pickle')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the individuals to the file
    with open(file_path, 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'.pickle', 'wb') as file:
    #     pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    # file.close()
    return

def save_PCdiversity_to_csv(randomSeeds, dataSetName, PCdiversity):
    """
    Save the PC diversity counts to a CSV file. Creates the file if it does not exist.

    :param PCdiversity: A Counter object with the number of occurrences of each unique PC set.
    :param filename: The name of the CSV file to write.
    """
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_PCdiversity.csv')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write data to the file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Gen', 'PCdiversity'])  # Write header
        # Assuming PCdiversity is a list of values; iterate and write to file
        for i in range(len(PCdiversity)):
            diversity = PCdiversity[i]
            writer.writerow([i, diversity])
    print(f"PC diversity data has been saved to {file_path}.")

    return

def save_each_gen_best_individual_meng(randomSeeds, dataSetName, best_ind_all_gen):
    individual_dict = {}

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)
        individual.append(sequencing_list)


        if len(best_ind) == 2:
            routing = best_ind[1]
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)
            individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_meng_individual_{dataSetName}.pkl')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the individual_dict to the file
    with open(file_path, 'wb') as fileName_individual:
        pickle.dump(individual_dict, fileName_individual, protocol=pickle.HIGHEST_PROTOCOL)

    return

def save_individual_to_txt(randomSeeds, dataSetName,individuals): # save individual as txt by mengxu
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}.txt')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write data to the file
    with open(file_path, 'w') as file:
        file.write('Individual:\n')
        file.write('Tree 0:\n')  # Routing rule
        file.write(str(individuals[0]) + '\n')
        if len(individuals) == 2:
            file.write('Tree 1:\n')  # Sequencing rule
            file.write(str(individuals[1]) + '\n')

    return

def clear_individual_each_gen_to_txt(randomSeeds, dataSetName): # save individual as txt by mengxu
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_each_gen.txt')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write data to the file
    with open(file_path, 'w') as file:
        file.write("Best individuals from each gen:\n")

    return

def save_individual_each_gen_to_txt(randomSeeds, dataSetName, individuals, gen): # save individual as txt by mengxu
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_each_gen.txt')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Append data to the file
    with open(file_path, 'a') as file:
        file.write('\nGen: ' + str(gen) + '\n')
        file.write('Individual:\n')
        file.write('Tree 0:\n')  # Routing rule
        file.write(str(individuals[0]) + '\n')
        if len(individuals) == 2:
            file.write('Tree 1:\n')  # Sequencing rule
            file.write(str(individuals[1]) + '\n')

    return


def save_archive(randomSeeds, dataSetName,individuals):
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_archive{dataSetName}.pickle')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the individuals to the file
    with open(file_path, 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)

    return

def save_pop(randomSeeds, dataSetName,individuals):
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_pop{dataSetName}.pickle')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the individuals to the file
    with open(file_path, 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)

    return


def saveMinFitness(randomSeeds, dataSetName, min_fitness):
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_min_fitness{dataSetName}.npy')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the min_fitness array to the file
    np.save(file_path, min_fitness)

    return

def saveRunningTime(randomSeeds, dataSetName, running_time):
    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/'
    file_path = os.path.join(directory, f'{randomSeeds}_running_time{dataSetName}.npy')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the running_time array to the file
    np.save(file_path, running_time)

    return

def save_TestResults_to_csv(randomSeeds, dataSetName, resultsDf):

    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/test/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_testResults.csv')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the DataFrame to a CSV file
    resultsDf.to_csv(file_path, index=False)

    return


def save_TestResults_final_gen_each_instance_to_csv(randomSeeds, dataSetName, resultsDf):

    # Construct the directory and file path
    directory = f'./MTGP_niching/train/scenario_{dataSetName}/test/'
    file_path = os.path.join(directory, f'{randomSeeds}_{dataSetName}_testResults_final_gen_each_instance.csv')

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the DataFrame to a CSV file
    resultsDf.to_csv(file_path, index=False)

    return