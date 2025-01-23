from operator import attrgetter
import random
from deap import tools
import numpy as np
from CCGP_niching_rental import saveFile
from CCGP_niching_rental.selection import selElitistAndTournament
from CCGP_niching_rental.niching.niching import niching_clear
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
                results = toolbox.mutate(offspring[i])
                if(len(results)>1):
                    print("Error here ere")
                offspring[i], = results
                # offspring[i], = toolbox.mutate(offspring[i])
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


def eaSimple(randomSeed_ngen, population, toolbox, cxpb, mutpb, reppb, elitism, ngen, seedRotate, use_niching, rd, stats=None,halloffame=None, verbose=__debug__, seed = __debug__, dataset_name=__debug__):
    # initialise the random seed of each generation
    # randomSeed_ngen = []
    # for i in range((ngen + 1)):
    # # for i in range((ngen+1)*ins_each_gen): # the *ins_each_gen is added by mengxu followed the advice of Meng 2022.11.01
    #     randomSeed_ngen.append(np.random.randint(2000000000))

    # get parameters for the given dataset/scenario
    scenarioDesign = ScenarioDesign_rental(dataset_name)
    parameters = scenarioDesign.get_parameter()

    logbook = tools.Logbook()
    logbook.header = ['gen', 'subpop', 'nevals'] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = [] #add by mengxu
    # Evaluate the individuals with an invalid fitness
    invalid_population = []
    for subpop in population:
        invalid_ind = [ind for ind in subpop if not ind.fitness.valid]
        invalid_population.append(invalid_ind)

    rd['seed'] = randomSeed_ngen[0]

    print("Instance seed: ", rd['seed'])

    fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_population, rd['seed'], parameters)
    for i in range(len(invalid_population)):
        invalid_ind = invalid_population[i]
        fitness_sub = fitnesses[i]
        for ind, fit in zip(invalid_ind, fitness_sub):
            ind.fitness.values = fit

    best_fit = np.inf
    best_combined_ind = []
    for i in range(len(population)):
        subpop = population[i]
        pop_fit = [ind.fitness.values[0] for ind in subpop]
        best_index_subpop = np.argmin(pop_fit)
        if(min(pop_fit)<best_fit):
            best_fit = min(pop_fit)
            best_index = best_index_subpop
        best_combined_ind.append(population[i][best_index_subpop])
    min_fitness.append(best_fit)
    best_ind_all_gen.append(best_combined_ind)  # add by mengxu
    p_one = best_combined_ind
    saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, 0)

    for i in range(len(population)):
        subpop = population[i]
        if halloffame is not None:
            halloffame.update(subpop)
        record = stats.compile(subpop) if stats else {}
        logbook.record(gen=0, subpop=i,  nevals=len(subpop), **record)
        if verbose:
            print(logbook.stream)

    # using niching to clear duplicated individual 2024.8.5
    PC_diversity_all = []
    nich = niching_clear(0, 1)
    nich.initial_phenoCharacterisation(parameters,best_combined_ind)
    population, PC_pop = nich.clearPopulation(toolbox, population, use_niching)
    # calculate the PC diversity of population
    calculator_subpop1 = PCDiversityCalculator(PC_pop[0])
    calculator_subpop2 = PCDiversityCalculator(PC_pop[1])
    PC_diversity_subpop1 = calculator_subpop1.calculate_diversity()
    PC_diversity_subpop2 = calculator_subpop2.calculate_diversity()
    PC_diversity = [PC_diversity_subpop1, PC_diversity_subpop2]
    print("PC diversity: " + str(PC_diversity))
    PC_diversity_all.append(PC_diversity)

    np.random.seed(seed) #add by mengxu to avoid niching make the same seed

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Added by mengxu to do seed rotation
        if seedRotate:
            rd['seed'] = randomSeed_ngen[gen]

        # Select the next generation individuals
        sorted_elite = []
        elitism_subpop = int(elitism / len(population))
        for subpop in population:
            sorted_elite_subpop = sortPopulation(toolbox, subpop)[:elitism_subpop]  # modified by mengxu 2022.10.29
            sorted_elite.append(sorted_elite_subpop)

        offspring = []
        for subpop in population:
            offspring_sub_pop= toolbox.select(subpop, len(subpop)-elitism_subpop)
            offspring.append(offspring_sub_pop)

        # Vary the pool of individuals
        # print('ori',offspring[0][0])
        # print('ori',offspring[0][1])
        # print('ori',offspring[0][2])
        for i in range(len(offspring)):
            offspring[i] = varAnd(offspring[i], toolbox, cxpb, mutpb, reppb)
        # print('after',offspring[0][0])
        # print('after',offspring[0][1])
        # print('after',offspring[0][2])
        # exit()

        # Evaluate the sorted_elite with an invalid fitness as we rotate seed, add by mengxu
        # invalid_elite_ind = [ind for ind in sorted_elite if not ind.fitness.valid]
        invalid_elite_pop = sorted_elite #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for invalid_elite_ind in invalid_elite_pop:
            for ind in invalid_elite_ind:
                del ind.fitness.values

        #combine elitism and offspring
        combined_offspring = []
        for i in range(len(offspring)):
            elitism_subpop = invalid_elite_pop[i]
            offspring_subpop = offspring[i]
            combined_subpop = elitism_subpop + offspring_subpop
            combined_offspring.append(combined_subpop)

        # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.multiProcess(toolbox.evaluate, combined_offspring, rd['seed'], parameters)
        for i in range(len(combined_offspring)):
            invalid_ind = combined_offspring[i]
            fitness_sub = fitnesses[i]
            for ind, fit in zip(invalid_ind, fitness_sub):
                ind.fitness.values = fit

        # Replace the current population by the offspring
        population[:] = combined_offspring
        # population[:] = sorted_elite+offspring

        # modified by mengxu
        for i in range(len(population)):
            subpop = population[i]
            if halloffame is not None:
                halloffame.clear()  # add by mengxu
                halloffame.update(subpop)
            # Append the current generation statistics to the logbook
            record = stats.compile(subpop) if stats else {}
            logbook.record(gen=gen, subpop=i, nevals=len(subpop), **record)
            if verbose:
                print(logbook.stream)

        best_fit = np.inf
        best_index = -1
        best_combined_ind = []
        for i in range(len(population)):
            subpop = population[i]
            pop_fit = [ind.fitness.values[0] for ind in subpop]
            best_index_subpop = np.argmin(pop_fit)
            if (min(pop_fit) < best_fit):
                best_fit = min(pop_fit)
                best_index = best_index_subpop
            best_combined_ind.append(population[i][best_index_subpop])
        min_fitness.append(best_fit)
        best_ind_all_gen.append(best_combined_ind)  # add by mengxu
        p_one = best_combined_ind
        saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, gen)

        # add by mengxu 2024.8.5 for niching---------------------------
        nich.calculate_phenoCharacterisation(best_combined_ind)
        population, PC_pop = nich.clearPopulation(toolbox, population, use_niching)
        # calculate the PC diversity of population
        calculator_subpop1 = PCDiversityCalculator(PC_pop[0])
        calculator_subpop2 = PCDiversityCalculator(PC_pop[1])
        PC_diversity_subpop1 = calculator_subpop1.calculate_diversity()
        PC_diversity_subpop2 = calculator_subpop2.calculate_diversity()
        PC_diversity = [PC_diversity_subpop1, PC_diversity_subpop2]
        print("PC diversity: " + str(PC_diversity))
        PC_diversity_all.append(PC_diversity)
        # print("PC diversity: " + str(PC_diversity))
        if gen == ngen:
            # save the PC diversity to a csv file
            saveFile.save_PCdiversity_to_csv(seed, dataset_name, PC_diversity_all)
        # add by mengxu 2024.8.5 for niching---------------------------

    return population, logbook, min_fitness, best_ind_all_gen

