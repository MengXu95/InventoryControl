from operator import attrgetter
import random
from deap import tools
import numpy as np
from MTGP_niching_rental import saveFile
from MTGP_niching_rental.selection import selElitistAndTournament
from MTGP_niching_rental.niching.niching import niching_clear
from Utils.PCDiversity import PCDiversityCalculator
from Utils.ScenarioDesign_rental import ScenarioDesign_rental

def varAnd(population, toolbox, cxpb, mutpb, reppb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb+reppb)
    new_mutpb=mutpb/(cxpb+mutpb+reppb)+new_cxpb
    i = 1
    while i < len(offspring):
        randomValue = random.random()
        if randomValue < new_cxpb: # crossover
            if (offspring[i - 1] == offspring[i]) :
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb: # mutation
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
        else: # reproduction
            del offspring[i].fitness.values
            i = i + 1
    return offspring

#todo: need to check if this is right
def sortPopulation(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)

    for j in range(popsize):
        sign = False
        for i in range(popsize-1-j):
            sum_fit_i = np.sum(populationCopy[i].fitness.values)
            sum_fit_i_1 = np.sum(populationCopy[i+1].fitness.values)
            if sum_fit_i > sum_fit_i_1:
                populationCopy[i], populationCopy[i+1] = populationCopy[i+1], populationCopy[i]
                sign = True
        if not sign:
            break

    #FOR CHECK
    # pop_fit = [np.sum(ind.fitness.values) for ind in
    #            populationCopy]
    # print(pop_fit)
    return populationCopy


def eaSimple(population, toolbox, cxpb, mutpb, reppb, elitism, ngen, seedRotate, use_niching, rd, stats=None,halloffame=None, verbose=__debug__, seed = __debug__, dataset_name=__debug__):
    # initialise the random seed of each generation
    randomSeed_ngen = []
    for i in range((ngen + 1)):
    # for i in range((ngen+1)*ins_each_gen): # the *ins_each_gen is added by mengxu followed the advice of Meng 2022.11.01
        randomSeed_ngen.append(np.random.randint(2000000000))

    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign_rental(dataset_name)
    parameters = scenarioDesign.get_parameter()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = [] #add by mengxu
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    rd['seed'] = randomSeed_ngen[0]

    fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_ind, rd['seed'], parameters)
    # fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_ind)
    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    pop_fit = [ind.fitness.values[0] for ind in population]
    min_fitness.append(min(pop_fit))
    # add by mengxu 2022.10.26
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index])  # add by mengxu
    p_one = population[best_index]
    saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, 0)

    if halloffame is not None:
        halloffame.update(population)

    # original
    # best_ind_all_gen.append(halloffame[0]) #add by mengxu
    # p_one = halloffame[0]
    # saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, 0)
    # best_ind_all_gen.append(halloffame[0])  # add by mengxu

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # using niching to clear duplicated individual 2024.8.5
    PC_diversity_all = []
    nich = niching_clear(0, 1)
    nich.initial_phenoCharacterisation(parameters, population[best_index])
    population, PC_pop = nich.clearPopulation(toolbox, population, use_niching)
    # calculate the PC diversity of population
    calculator = PCDiversityCalculator(PC_pop)
    PC_diversity = calculator.calculate_diversity()
    PC_diversity_all.append(PC_diversity)
    print("PC diversity: " + str(PC_diversity))

    np.random.seed(seed) #add by mengxu to avoid niching make the same seed

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Added by mengxu to do seed rotation
        if seedRotate:
            rd['seed'] = randomSeed_ngen[gen]
            # rd['seed'] = np.random.randint(2000000000)
        # Select the next generation individuals
        sorted_elite = sortPopulation(toolbox, population)[:elitism]  # modified by mengxu 2022.10.29
        # sorted_elite = sorted(population, key=attrgetter("fitness"), reverse=True)[:elitism]

        offspring = toolbox.select(population, len(population)-elitism)

        # Vary the pool of individuals
        # print('ori',offspring[0][0])
        # print('ori',offspring[0][1])
        # print('ori',offspring[0][2])
        offspring = varAnd(offspring, toolbox, cxpb, mutpb, reppb)
        # print('after',offspring[0][0])
        # print('after',offspring[0][1])
        # print('after',offspring[0][2])
        # exit()

        # Evaluate the sorted_elite with an invalid fitness as we rotate seed, add by mengxu
        # invalid_elite_ind = [ind for ind in sorted_elite if not ind.fitness.valid]
        invalid_elite_ind = sorted_elite #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for ind in invalid_elite_ind:
            del ind.fitness.values

        fitnesses_elite = toolbox.multiProcess(toolbox.evaluate, invalid_elite_ind, rd['seed'], parameters)
        # fitnesses_elite = toolbox.map(toolbox.evaluate, invalid_elite_ind)
        for ind, fit in zip(invalid_elite_ind, fitnesses_elite):
            ind.fitness.values = fit

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = offspring #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for ind in invalid_ind:
            del ind.fitness.values


        fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_ind, rd['seed'], parameters)
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        # if halloffame is not None:
        #     halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = invalid_elite_ind + invalid_ind
        # population[:] = sorted_elite+offspring

        # modified by mengxu
        if halloffame is not None:
            halloffame.clear()  # add by mengxu
            halloffame.update(population)
            # sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)
            # halloffame.insert(sorted_pop[0])

        # for store each generation best individual add by mengxu
        # dataset_name = 'dataSet_DFJSS'
        # print(seed)

        # add by mengxu 2022.10.26
        pop_fit = [ind.fitness.values[0] for ind in population]
        best_index = np.argmin(pop_fit)
        best_ind_all_gen.append(population[best_index])  # add by mengxu
        p_one = population[best_index]
        saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, gen)


        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        # add by mengxu 2024.8.5 for niching---------------------------
        nich.calculate_phenoCharacterisation(population[best_index])
        population, PC_pop = nich.clearPopulation(toolbox, population, use_niching)
        # calculate the PC diversity of population
        calculator = PCDiversityCalculator(PC_pop)
        PC_diversity = calculator.calculate_diversity()
        PC_diversity_all.append(PC_diversity)
        print("PC diversity: " + str(PC_diversity))
        if gen == ngen:
            # save the PC diversity to a csv file
            saveFile.save_PCdiversity_to_csv(seed, dataset_name, PC_diversity_all)
        # add by mengxu 2024.8.5 for niching---------------------------

        pop_fit = [ind.fitness.values[0] for ind in population]  ######selection from author
        min_fitness.append(min(pop_fit))
    return population, logbook, min_fitness, best_ind_all_gen

