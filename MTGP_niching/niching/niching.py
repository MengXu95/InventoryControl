import MTGP_niching.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import MTGP_niching.replenishment as sequencing
import MTGP_niching.niching.ReplenishmentPhenoCharacterisation as ReplenishmentPhenoCharacterisation
import MTGP_niching.niching.TransshipmentPhenoCharacterisation as TransshipmentPhenoCharacterisation
from MTGP_niching.niching.Inventory_simulator_niching import *

class simulator_niching:
    def __init__(self, num_decision_points, individual, **kwargs):
        '''STEP 1: create environment instances!'''
        self.inventory_env = InvOptEnv(77777)
        self.num_decision_points = num_decision_points
        self.decision_points = self.inventory_env.run_to_get_decision(individual)

    def getDecisionSituations(self):
        decisionSituations = []
        if len(self.decision_points) == 1: # only consider replenishment
            replenishment_decision_points = self.decision_points[0]
            if len(replenishment_decision_points) < self.num_decision_points:
                print("Error in get enough number of self.replenishment_decision_points)")

            np.random.shuffle(replenishment_decision_points)
            subset_replenishment_decision_points = replenishment_decision_points[:20]

            decisionSituations.append(subset_replenishment_decision_points)
        elif len(self.decision_points) == 2: # consider both replenishment and transshipment
            replenishment_decision_points = self.decision_points[0]
            if len(replenishment_decision_points) < self.num_decision_points:
                print("Error in get enough number of self.replenishment_decision_points)")

            np.random.shuffle(replenishment_decision_points)
            subset_replenishment_decision_points = replenishment_decision_points[:20]
            decisionSituations.append(subset_replenishment_decision_points)

            transshipment_decision_points = self.decision_points[1]
            if len(transshipment_decision_points) < self.num_decision_points:
                print("Error in get enough number of self.replenishment_decision_points)")

            np.random.shuffle(transshipment_decision_points)
            subset_transshipment_decision_points = transshipment_decision_points[:20]
            decisionSituations.append(subset_transshipment_decision_points)
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
        elif len(individual) == 2: # for the instances that consider both the replenishment and transshipment
            replenishmentDecisionSituation = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(
                self.decisionSituations[0])
            replenishmentPhenoCharacterisation = ReplenishmentPhenoCharacterisation.ReplenishmentPhenoCharacterisation(
                individual[0], replenishmentDecisionSituation)
            self.phenotypic_characristics.append(replenishmentPhenoCharacterisation)
            transshipmentDecisionSituation = TransshipmentDecisionSituation.TransshipmentDecisionSituation(self.decisionSituations[1])
            transshipmentPhenoCharacterisation = TransshipmentPhenoCharacterisation.TransshipmentPhenoCharacterisation(individual[1], transshipmentDecisionSituation)
            self.phenotypic_characristics.append(transshipmentPhenoCharacterisation)
        else:
            # this is for also consider transshipment
            print('Error in niching!')


    def calculate_phenoCharacterisation(self, individual):
        # create the environment instance for simulation
        if len(individual) == 2:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])
            self.phenotypic_characristics[1].setReferenceRule(individual[1])
        else:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])


    def clearPopulation(self,toolbox,population):
        if len(population[0]) == 1: # consider only replenishment
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
        elif len(population[0]) == 2: # consider both replenishment and transshipment
            clearedInds = 0
            phenotypic_characristics_pop = []
            sorted_pop = self.sortPopulation(toolbox, population)
            isCleared_pop = []
            # calculate the PC of all individuals in population
            for idx in range(len(sorted_pop)):
                ind = sorted_pop[idx]
                replenishment_charList = self.phenotypic_characristics[0].characterise(ind[0])
                transshipment_charList = self.phenotypic_characristics[1].characterise(ind[1])
                all_charList = []
                for ref in replenishment_charList:
                    all_charList.append(ref)
                for ref in transshipment_charList:
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
            print("Error in niching!")
        return sorted_pop, phenotypic_characristics_pop


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