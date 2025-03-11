import numpy as np


class niching_clear:
    def __init__(self, radius, capacity, **kwargs):
        self.radius = radius
        self.capacity = capacity

    def clearPopulation(self,toolbox,population, Clear=True):
        clearedInds = 0
        fitness_pop = []
        sorted_pop = self.sortPopulation(toolbox, population)
        isCleared_pop = []
        # calculate the PC of all individuals in population
        for idx in range(len(sorted_pop)):
            ind = sorted_pop[idx]
            fitness = np.mean(ind.fitness.values)
            fitness_pop.append(fitness)
            isCleared_pop.append(False)

        if Clear:
            # clear this population
            for idx in range(len(sorted_pop)):
                if isCleared_pop[idx]:
                    continue

                numWinners = 1
                for idy in range(idx + 1, len(sorted_pop)):
                    if isCleared_pop[idy]:
                        continue

                    fit_idx = np.mean(sorted_pop[idx].fitness.values)
                    fit_idy = np.mean(sorted_pop[idy].fitness.values)
                    distance = self.distance(fit_idx, fit_idy)
                    if distance > self.radius:
                        continue

                    if numWinners < self.capacity:
                        numWinners = numWinners + 1
                    else:
                        isCleared_pop[idy] = True
                        len_fitness_values = len(sorted_pop[idy].fitness.values)
                        bad_fitness = [np.Infinity for i in range(len_fitness_values)]
                        sorted_pop[idy].fitness.values = bad_fitness
                        clearedInds = clearedInds + 1
            print("Cleared number by niching: " + str(clearedInds))
        return sorted_pop, fitness_pop


    def sortPopulation(self, toolbox, population):
        populationCopy = [toolbox.clone(ind) for ind in population]
        popsize = len(population)

        for j in range(popsize):
            sign = False
            for i in range(popsize - 1 - j):
                sum_fit_i = np.sum(populationCopy[i].fitness.values)
                sum_fit_i_1 = np.sum(populationCopy[i + 1].fitness.values)
                if sum_fit_i > sum_fit_i_1:
                    populationCopy[i], populationCopy[i + 1] = populationCopy[i + 1], populationCopy[i]
                    sign = True
            if not sign:
                break

        # FOR CHECK
        # pop_fit = [np.sum(ind.fitness.values) for ind in
        #            populationCopy]
        # print(pop_fit)
        return populationCopy

    def distance(self, valueA, valueB):
        return np.sqrt((valueA-valueB)*(valueA-valueB))
