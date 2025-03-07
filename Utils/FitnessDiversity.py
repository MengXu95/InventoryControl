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



# Example usage
if __name__ == "__main__":
    population = [
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.0, 2.0, 3.0],
    ]

    calculator = PCDiversityCalculator(population)
    PC_diversity = calculator.calculate_diversity()

    print(f"PC diversity: {PC_diversity}")
