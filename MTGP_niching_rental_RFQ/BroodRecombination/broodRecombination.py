import numpy as np

from MTGP_niching_rental_RFQ.niching.niching import simulator_niching, niching_clear
from Utils.ScenarioDesign_rental_RFQ import ScenarioDesign_rental_RFQ

class broodPop:
    def __init__(self, original_size, dataset_name, referenceInd, brood_times=5, **kwargs):
        self.brood_size = original_size * brood_times
        self.original_size = original_size
        self.nich = None
        self.initialReferencePoint(dataset_name, referenceInd)
        self.keyRegionRadius = np.inf
        self.threshold = np.inf

    def initialReferencePoint(self, dataset_name, referenceInd):
        # get parameters for the given dataset/scenario
        scenarioDesign = ScenarioDesign_rental_RFQ(dataset_name)
        parameters = scenarioDesign.get_parameter()

        self.nich = niching_clear(0, 1)
        self.nich.initial_phenoCharacterisation(parameters, referenceInd)

    def shrinkPopToSize(self, population, size=None):
        bestInd = population[0]
        self.nich.calculate_phenoCharacterisation(bestInd)
        population_dis = []

        for ind in population:
            replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
            rental_charList = self.nich.phenotypic_characristics[1].characterise(ind[1])
            RFQ_predict_charList = self.nich.phenotypic_characristics[2].characterise(ind[2])

            # Will ignore rental cause rental decisions are always feasible,
            # but replenishment and RFQ predict might not be feasible
            replenishment_dis = self.nich.phenotypic_characristics[0].distance(replenishment_charList,
                                                                               self.nich.phenotypic_characristics[0].decisions)
            RFQ_predict_dis = self.nich.phenotypic_characristics[2].distance(RFQ_predict_charList,
                                                                               self.nich.phenotypic_characristics[2].decisions)
            population_dis.append(replenishment_dis+RFQ_predict_dis)

        unique_population, unique_population_dis, other_population = self.sortPopBasedonPopDis(population, population_dis)

        print("unique_population length: ", len(unique_population))
        print("unique_population_dis length: ", len(unique_population_dis))

        if size is not None:
            if len(unique_population) >= size:
                new_pop = unique_population[:size]
                self.keyRegionRadius = unique_population_dis[size-1]
            else:
                new_pop = unique_population
                self.keyRegionRadius = unique_population_dis[len(unique_population) - 1]
                while len(new_pop) < size:
                    index = np.random.randint(len(other_population))
                    new_pop.append(other_population[index])
        else:
            if len(unique_population) >= self.original_size:
                new_pop = unique_population[:self.original_size]
                self.keyRegionRadius = unique_population_dis[self.original_size - 1]
            else:
                new_pop = unique_population
                self.keyRegionRadius = unique_population_dis[len(unique_population) - 1]
                while len(new_pop) < self.original_size:
                    index = np.random.randint(len(other_population))
                    new_pop.append(other_population[index])
        print("key Region Radius: ", self.keyRegionRadius)
        return new_pop

    def adjustThreshold(self, lower_bound_dis, upper_bound_dis):
        gap = upper_bound_dis - lower_bound_dis

        if gap < 1e-3:  # If gap is too small, keep the threshold stable
            self.threshold = (self.threshold + lower_bound_dis + upper_bound_dis) / 3
        else:
            scaling_factor = min(1, max(0.1, 1 - (1 / (1 + gap))))  # Adjust scaling based on gap
            new_threshold = lower_bound_dis + scaling_factor * (upper_bound_dis - lower_bound_dis)

            # Smooth transition from previous threshold to avoid sudden changes
            alpha = 0.7  # Weight for previous threshold (adjustable)
            self.threshold = alpha * self.threshold + (1 - alpha) * new_threshold


    def shrinkPopToSizeBasedOnRadius(self, population, size=None):
        bestInd = population[0]
        self.nich.calculate_phenoCharacterisation(bestInd)
        population_dis = []

        for ind in population:
            replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
            rental_charList = self.nich.phenotypic_characristics[1].characterise(ind[1])
            RFQ_predict_charList = self.nich.phenotypic_characristics[2].characterise(ind[2])

            # Will ignore rental cause rental decisions are always feasible,
            # but replenishment and RFQ predict might not be feasible
            replenishment_dis = self.nich.phenotypic_characristics[0].distance(replenishment_charList,
                                                                               self.nich.phenotypic_characristics[0].decisions)
            RFQ_predict_dis = self.nich.phenotypic_characristics[2].distance(RFQ_predict_charList,
                                                                               self.nich.phenotypic_characristics[2].decisions)
            population_dis.append(replenishment_dis+RFQ_predict_dis)

        unique_population, unique_population_dis, other_population = self.sortPopBasedonPopDis(population, population_dis)


        self.adjustThreshold(unique_population_dis[0], unique_population_dis[-1])

        clip_index = len(unique_population_dis)
        for i in range(len(unique_population_dis)-1):
            dis = unique_population_dis[i]
            dis_next = unique_population_dis[i+1]
            if self.threshold > unique_population_dis[0] and dis < self.threshold and dis_next > self.threshold:
                clip_index = i+1
                break

        unique_population, unique_population_dis = unique_population[:clip_index], unique_population_dis[:clip_index]
        other_population = other_population + unique_population[clip_index:]
        print("unique_population length: ", len(unique_population))
        print("unique_population_dis length: ", len(unique_population_dis))

        if size is not None:
            if len(unique_population) >= size:
                np.random.shuffle(unique_population)
                new_pop = unique_population[:size]
                self.keyRegionRadius = unique_population_dis[size-1]
            else:
                new_pop = unique_population
                self.keyRegionRadius = unique_population_dis[len(unique_population) - 1]
                while len(new_pop) < size:
                    index = np.random.randint(len(other_population))
                    new_pop.append(other_population[index])
        else:
            if len(unique_population) >= self.original_size:
                np.random.shuffle(unique_population)
                new_pop = unique_population[:self.original_size]
                self.keyRegionRadius = unique_population_dis[self.original_size - 1]
            else:
                new_pop = unique_population
                self.keyRegionRadius = unique_population_dis[len(unique_population) - 1]
                while len(new_pop) < self.original_size:
                    index = np.random.randint(len(other_population))
                    new_pop.append(other_population[index])
        print("key Region Radius: ", self.keyRegionRadius)
        return new_pop


    def sortPopBasedonPopDis(self, population, population_dis):
        # Combine population and population_dis into tuples
        combined = list(zip(population, population_dis))

        # Sort based on population_dis (second element of the tuple)
        combined.sort(key=lambda x: x[1])

        # Unzip the sorted tuples back into separate lists
        population_sorted, population_dis_sorted = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        population_sorted, population_dis_sorted = list(population_sorted), list(population_dis_sorted)

        # Remove duplicates based on population_dis_sorted
        return self.removeDuplicates(population_sorted, population_dis_sorted)

    def removeDuplicates(self, population_sorted, population_dis_sorted):
        unique_population = []
        unique_population_dis = []
        other_population = []
        seen_dis = set()

        for ind, dis in zip(population_sorted, population_dis_sorted):
            if dis not in seen_dis:
                seen_dis.add(dis)
                unique_population.append(ind)
                unique_population_dis.append(dis)
            else:
                other_population.append(ind)

        return unique_population, unique_population_dis, other_population
