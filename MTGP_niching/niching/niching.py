import MTGP_niching.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import MTGP_niching.sequencing as sequencing
import MTGP_niching.niching.ReplenishmentPhenoCharacterisation as ReplenishmentPhenoCharacterisation
from MTGP_niching.niching.Inventory_simulator_niching import *

class simulator_niching:
    def __init__(self, num_decision_points, individual, **kwargs):
        '''STEP 1: create environment instances!'''
        self.inventory_env = InvOptEnv(77777)
        self.num_decision_points = num_decision_points
        self.replenishment_decision_points = self.inventory_env.run_to_get_decision(individual)

    def getDecisionSituations(self):
        decisionSituations = []
        if len(self.replenishment_decision_points) < 20:
            print("Error in get enough number of self.replenishment_decision_points)")

        np.random.shuffle(self.replenishment_decision_points)
        subset_replenishment_decision_points = self.replenishment_decision_points[:20]

        decisionSituations.append(subset_replenishment_decision_points)
        # decisionSituations.append(subset_routingDecisionSituations) # this is for considering the transshipment value
        return decisionSituations


class niching_clear:
    def __init__(self, radius, capacity, **kwargs):
        self.radius = radius
        self.capacity = capacity
        self.decisionSituations = []
        self.phenotypic_characristics = []


    def initial_phenoCharacterisation(self, individual):
        # create the environment instance for simulation
        self.phenotypic_characristics = []
        env_niching = simulator_niching(20, individual)
        self.decisionSituations = env_niching.getDecisionSituations()
        if len(individual) == 1: # for the instances that only consider the replenishment rule
            replenishmentDecisionSituation = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(self.decisionSituations[0])
            replenishmentPhenoCharacterisation = ReplenishmentPhenoCharacterisation.ReplenishmentPhenoCharacterisation(individual[0], replenishmentDecisionSituation)
            self.phenotypic_characristics.append(replenishmentPhenoCharacterisation)
        else:
            # this is for also consider transshipment
            print('Need to implement this when considering transshipment!')


    def calculate_phenoCharacterisation(self, individual):
        # create the environment instance for simulation
        if len(individual) == 2:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])
            self.phenotypic_characristics[1].setReferenceRule(individual[1])
        else:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])


    def clearPopulation(self,toolbox,population):
        if len(population[0]) == 1:
            clearedInds = 0
            phenotypic_characristics_pop = []
            sorted_pop = self.sortPopulation(toolbox, population)
            isCleared_pop = []
            # calculate the PC of all individuals in population
            for idx in range(len(sorted_pop)):
                ind = sorted_pop[idx]
                replenishment_charList = self.phenotypic_characristics[0].characterise(ind[0])
                all_charList = []
                for ref in replenishment_charList:
                    all_charList.append(ref)
                phenotypic_characristics_pop.append(all_charList)
                isCleared_pop.append(False)

            #clear this population
            for idx in range(len(sorted_pop)):
                if isCleared_pop[idx]:
                    continue

                numWinners = 1
                for idy in range(idx+1, len(sorted_pop)):
                    if isCleared_pop[idy]:
                        continue

                    distance = self.phenotypic_characristics[0].distance(
                        phenotypic_characristics_pop[idx], phenotypic_characristics_pop[idy])
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
        else:
            # this is for considering transshipment
            print("Need to implement this for considering transshipment!")
        return sorted_pop


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