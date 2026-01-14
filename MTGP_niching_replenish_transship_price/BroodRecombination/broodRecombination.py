import numpy as np
import statistics

from MTGP_niching_replenish_transship_price.niching.niching import simulator_niching, niching_clear
from Utils.ScenarioDesign_replenish_transship_price import ScenarioDesign_replenish_transship_price


class broodPop:
    def __init__(self, original_size, dataset_name, referenceInd, brood_times=5, **kwargs):
        self.brood_size = original_size * brood_times
        self.original_size = original_size
        self.nich = None
        self.initialReferencePoint(dataset_name, referenceInd)
        self.keyRegionRadius = np.inf
        self.threshold = np.inf
        self.MAX_VALUE = np.inf

    def initialReferencePoint(self, dataset_name, referenceInd):
        # get parameters for the given dataset/scenario
        scenarioDesign = ScenarioDesign_replenish_transship_price(dataset_name)
        parameters = scenarioDesign.get_parameter()

        self.nich = niching_clear(0, 1)
        self.nich.initial_phenoCharacterisation(parameters, referenceInd)

    def shrinkPopToSize(self, population, size=None):
        bestInd = population[0]
        self.nich.calculate_phenoCharacterisation(bestInd)
        new_population = []
        new_population_dis = []

        for ind in population:
            if len(ind) == 1:
                # Only replenishment
                replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
                replenishment_dis = self.nich.phenotypic_characristics[0].distance(
                    replenishment_charList, self.nich.phenotypic_characristics[0].decisions)
                total_dis = replenishment_dis / len(replenishment_charList)

            elif len(ind) == 2:
                # Replenishment + Transshipment
                replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
                transshipment_charList = self.nich.phenotypic_characristics[1].characterise(ind[1])

                replenishment_dis = self.nich.phenotypic_characristics[0].distance(
                    replenishment_charList, self.nich.phenotypic_characristics[0].decisions)
                transshipment_dis = self.nich.phenotypic_characristics[1].distance(
                    transshipment_charList, self.nich.phenotypic_characristics[1].decisions)

                total_dis = (replenishment_dis + transshipment_dis) / (
                        len(replenishment_charList) + len(transshipment_charList))

            elif len(ind) == 3:
                # Replenishment + Transshipment + RFQ Predict
                replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
                transshipment_charList = self.nich.phenotypic_characristics[1].characterise(ind[1])
                RFQ_predict_charList = self.nich.phenotypic_characristics[2].characterise(ind[2])

                replenishment_dis = self.nich.phenotypic_characristics[0].distance(
                    replenishment_charList, self.nich.phenotypic_characristics[0].decisions)
                transshipment_dis = self.nich.phenotypic_characristics[1].distance(
                    transshipment_charList, self.nich.phenotypic_characristics[1].decisions)
                RFQ_predict_dis = self.nich.phenotypic_characristics[2].distance(
                    RFQ_predict_charList, self.nich.phenotypic_characristics[2].decisions)

                total_dis = (replenishment_dis + transshipment_dis + RFQ_predict_dis) / (
                        len(replenishment_charList) + len(transshipment_charList) + len(RFQ_predict_charList))
            else:
                print("Error in brood recombination!")
                continue

            if total_dis < self.MAX_VALUE:
                new_population.append(ind)
                new_population_dis.append(total_dis)

        unique_population, unique_population_dis, other_population = self.sortPopBasedonPopDis(
            new_population, new_population_dis)

        if size is not None:
            if len(unique_population) >= size:
                new_pop = unique_population[:size]
                self.keyRegionRadius = unique_population_dis[size - 1]
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

        return new_pop

    def adjustThreshold(self, unique_population_dis):
        """
        Adjusts the threshold for clipping top-ranked individuals based on their distance distribution.

        Rationale:
            - Handles cases with both small and large variations in distances.
            - Uses a robust measure of dispersion (Median Absolute Deviation, MAD)
              instead of the median of absolute differences.
            - Adjusts the threshold relative to the minimum distance, ensuring it's
              always a meaningful value.
            - Introduces a scaling factor to control the threshold's sensitivity.

        Args:
            unique_population_dis: A list or NumPy array of distances.

        Returns:
            None. Sets self.threshold.
        """

        unique_population_dis = np.array(unique_population_dis)
        if not unique_population_dis.size:  # Handle empty list case
            self.threshold = 0.0
            return

        lower_bound_dis = np.min(unique_population_dis)

        # Calculate Median Absolute Deviation (MAD)
        median_val_original = np.median(unique_population_dis)
        absolute_deviations = np.abs(unique_population_dis - median_val_original)
        mad = np.median(absolute_deviations)

        # Scaling factor (adjust as needed)
        scaling_factor = 2  # Controls how much dispersion is considered

        # Calculate the threshold
        self.threshold = lower_bound_dis + scaling_factor * mad
        self.keyRegionRadius = scaling_factor * mad

    def shrinkPopToSizeBasedOnRadius(self, population, size=None):
        bestInd = population[0]
        self.nich.calculate_phenoCharacterisation(bestInd)
        new_population = []
        new_population_dis = []

        for ind in population:
            if len(ind) == 1:
                # Only replenishment
                replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
                replenishment_dis = self.nich.phenotypic_characristics[0].distance(
                    replenishment_charList, self.nich.phenotypic_characristics[0].decisions)
                total_dis = replenishment_dis / len(replenishment_charList)

            elif len(ind) == 2:
                # Replenishment + Transshipment
                replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
                transshipment_charList = self.nich.phenotypic_characristics[1].characterise(ind[1])

                # Calculate distance for both replenishment and transshipment
                replenishment_dis = self.nich.phenotypic_characristics[0].distance(
                    replenishment_charList, self.nich.phenotypic_characristics[0].decisions)
                transshipment_dis = self.nich.phenotypic_characristics[1].distance(
                    transshipment_charList, self.nich.phenotypic_characristics[1].decisions)

                total_dis = (replenishment_dis + transshipment_dis) / (
                        len(replenishment_charList) + len(transshipment_charList))

            elif len(ind) == 3:
                # Replenishment + Transshipment + RFQ Predict
                replenishment_charList = self.nich.phenotypic_characristics[0].characterise(ind[0])
                transshipment_charList = self.nich.phenotypic_characristics[1].characterise(ind[1])
                RFQ_predict_charList = self.nich.phenotypic_characristics[2].characterise(ind[2])

                # Calculate distances for replenishment, transshipment, and RFQ predict
                replenishment_dis = self.nich.phenotypic_characristics[0].distance(
                    replenishment_charList, self.nich.phenotypic_characristics[0].decisions)
                transshipment_dis = self.nich.phenotypic_characristics[1].distance(
                    transshipment_charList, self.nich.phenotypic_characristics[1].decisions)
                RFQ_predict_dis = self.nich.phenotypic_characristics[2].distance(
                    RFQ_predict_charList, self.nich.phenotypic_characristics[2].decisions)

                total_dis = (replenishment_dis + transshipment_dis + RFQ_predict_dis) / (
                        len(replenishment_charList) + len(transshipment_charList) + len(RFQ_predict_charList))
            else:
                print("Error in brood recombination!")
                continue

            if total_dis < self.MAX_VALUE:
                new_population.append(ind)
                new_population_dis.append(total_dis)

        removeDup = True
        unique_population, unique_population_dis, other_population = self.sortPopBasedonPopDis(
            new_population, new_population_dis, removeDup)

        adjustThres = False
        if adjustThres:
            self.adjustThreshold(unique_population_dis)
        else:
            self.threshold = np.inf

        clip_index = len(unique_population_dis)

        # Find the index where the threshold is crossed
        for i in range(len(unique_population_dis) - 1):
            if self.threshold > unique_population_dis[0] and unique_population_dis[i] <= self.threshold < \
                    unique_population_dis[i + 1]:
                clip_index = i + 1
                break

        # Compute and display the clip percentage
        clip_percentage = clip_index / len(unique_population_dis) if len(unique_population_dis) > 0 else 0

        # Split the population based on the determined clip index
        new_unique_population = unique_population[:clip_index]
        new_unique_population_dis = unique_population_dis[:clip_index]
        other_unique_population = unique_population[clip_index:]

        if size is not None:
            if len(new_unique_population) >= size:
                np.random.shuffle(new_unique_population)
                new_pop = new_unique_population[:size]
            elif len(new_unique_population) + len(other_unique_population) >= size:
                np.random.shuffle(other_unique_population)
                new_pop = new_unique_population + other_unique_population[:(size - len(new_unique_population))]
            else:
                new_pop = new_unique_population
                new_pop = new_pop + other_unique_population
                np.random.shuffle(other_population)
                new_pop = new_pop + other_population[:(size - len(new_pop))]
        else:
            if len(new_unique_population) >= self.original_size:
                np.random.shuffle(new_unique_population)
                new_pop = new_unique_population[:self.original_size]
            elif len(new_unique_population) + len(other_unique_population) >= self.original_size:
                np.random.shuffle(other_unique_population)
                new_pop = new_unique_population + other_unique_population[
                    :(self.original_size - len(new_unique_population))]
            else:
                new_pop = new_unique_population
                new_pop = new_pop + other_unique_population
                np.random.shuffle(other_population)
                new_pop = new_pop + other_population[:(self.original_size - len(new_pop))]

        return new_pop

    def sortPopBasedonPopDis(self, population, population_dis, removeDup=True):
        # Combine population and population_dis into tuples
        combined = list(zip(population, population_dis))

        # Sort based on population_dis (second element of the tuple)
        combined.sort(key=lambda x: x[1])

        # Unzip the sorted tuples back into separate lists
        population_sorted, population_dis_sorted = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        population_sorted, population_dis_sorted = list(population_sorted), list(population_dis_sorted)

        # Remove duplicates based on population_dis_sorted
        if removeDup:
            return self.removeDuplicates(population_sorted, population_dis_sorted)
        else:
            other_population = []
            return population_sorted, population_dis_sorted, other_population

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