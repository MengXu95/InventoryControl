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
        subpop1 = population[0] # this is for replenishment
        subpop2 = population[1] # this is for transshipment

        # clear subpop1
        clearedInds_subpop1 = 0
        phenotypic_characristics_pop_subpop1 = []
        sorted_pop_subpop1 = self.sortPopulation(toolbox, subpop1)
        isCleared_pop_subpop1 = []
        # calculate the PC of all individuals in population
        for idx in range(len(sorted_pop_subpop1)):
            ind = sorted_pop_subpop1[idx]
            replenishment_charList = self.phenotypic_characristics[0].characterise(ind)
            all_charList = []
            for ref in replenishment_charList:
                all_charList.append(ref)
            phenotypic_characristics_pop_subpop1.append(all_charList)
            isCleared_pop_subpop1.append(False)

        # clear this population
        for idx in range(len(sorted_pop_subpop1)):
            if isCleared_pop_subpop1[idx]:
                continue

            numWinners = 1
            for idy in range(idx + 1, len(sorted_pop_subpop1)):
                if isCleared_pop_subpop1[idy]:
                    continue

                distance = self.phenotypic_characristics[0].distance(
                    phenotypic_characristics_pop_subpop1[idx], phenotypic_characristics_pop_subpop1[idy])
                if distance > self.radius:
                    continue

                if numWinners < self.capacity:
                    numWinners = numWinners + 1
                else:
                    isCleared_pop_subpop1[idy] = True
                    len_fitness_values = len(sorted_pop_subpop1[idy].fitness.values)
                    bad_fitness = [np.Infinity for i in range(len_fitness_values)]
                    sorted_pop_subpop1[idy].fitness.values = bad_fitness
                    clearedInds_subpop1 = clearedInds_subpop1 + 1
        print("Cleared number by niching for subpop1: " + str(clearedInds_subpop1))

        # clear subpop2
        clearedInds_subpop2 = 0
        phenotypic_characristics_pop_subpop2 = []
        sorted_pop_subpop2 = self.sortPopulation(toolbox, subpop2)
        isCleared_pop_subpop2 = []
        # calculate the PC of all individuals in population
        for idx in range(len(sorted_pop_subpop2)):
            ind = sorted_pop_subpop2[idx]
            ransshipment_charListt = self.phenotypic_characristics[1].characterise(ind)
            all_charList = []
            for ref in ransshipment_charListt:
                all_charList.append(ref)
            phenotypic_characristics_pop_subpop2.append(all_charList)
            isCleared_pop_subpop2.append(False)

        # clear this population
        for idx in range(len(sorted_pop_subpop2)):
            if isCleared_pop_subpop2[idx]:
                continue

            numWinners = 1
            for idy in range(idx + 1, len(sorted_pop_subpop2)):
                if isCleared_pop_subpop2[idy]:
                    continue

                distance = self.phenotypic_characristics[1].distance(
                    phenotypic_characristics_pop_subpop2[idx], phenotypic_characristics_pop_subpop2[idy])
                if distance > self.radius:
                    continue

                if numWinners < self.capacity:
                    numWinners = numWinners + 1
                else:
                    isCleared_pop_subpop2[idy] = True
                    len_fitness_values = len(sorted_pop_subpop2[idy].fitness.values)
                    bad_fitness = [np.Infinity for i in range(len_fitness_values)]
                    sorted_pop_subpop1[idy].fitness.values = bad_fitness
                    clearedInds_subpop2 = clearedInds_subpop2 + 1
        print("Cleared number by niching for subpop2: " + str(clearedInds_subpop2))
        cleared_pop = [sorted_pop_subpop1, sorted_pop_subpop2]
        PC_pop = [phenotypic_characristics_pop_subpop1, phenotypic_characristics_pop_subpop2]
        return cleared_pop, PC_pop


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