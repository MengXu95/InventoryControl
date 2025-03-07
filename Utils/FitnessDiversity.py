from collections import Counter
import csv
import os

class FitnessDiversityCalculator:
    def __init__(self, populationPC):
        """
        Initialize the PCDiversityCalculator with a population of individuals.

        :param population: List of individuals, where each individual is a list of PC values.
        """
        self.population = populationPC

    def calculate_diversity(self):
        """
        Calculate the PC diversity by counting how many individuals have the same PC values.

        :return: A dictionary with the number of occurrences of each unique PC set and
                 the number of unique individuals.
        """
        # Count the occurrences of each unique PC value set
        counts = Counter(self.population)

        # Calculate the number of unique individuals
        num_unique_individuals = len(counts)

        # Return the PC diversity
        return num_unique_individuals/len(self.population)




