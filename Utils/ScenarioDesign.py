import sys
import re

sys.path

class ScenarioDesign():
    def __init__(self, scenario):
        """
        Initialize the PCDiversityCalculator with a population of individuals.

        :param population: List of individuals, where each individual is a list of PC values.
        """
        self.scenario = scenario

    def get_parameter(self):
        # Assuming the string format is: "sN<retailers>h_<holding1>_<holding2>_<holding3>b<LT>"

        # Splitting the string
        parts = re.split('[Nhb]', self.scenario)

        # Extracting demand level of retailers
        demand_scale = parts[0]
        if demand_scale == "s":
            demand_level = 20
        elif demand_scale == "m":
            demand_level = 100
        else:
            demand_level = 1000

        # Extracting number of retailers (after "sN")
        num_retailer = int(parts[1])

        # Extracting holding costs
        str_holding = parts[2].split('_')[1:]

        # Convert to a list of integers
        holding = [int(item) for item in str_holding]

        # Extracting lead time (after "b")
        b = int(parts[3])

        # Calculated values
        L = 2  # Length of forecast horizon (default)
        LT = 2
        epi_len = 64  # Length of one episode (default)
        ini_inv = [10] * num_retailer  # Initial inventory levels
        lost_sales = [b * h for h in holding]  # Per unit lost sales costs
        capacity = [5 * demand_level] * num_retailer  # Inventory capacities
        fixed_order = [20] * num_retailer  # Fixed order costs per order
        per_trans_item = 1  # Per unit cost for transshipment (either direction)
        per_trans_order = 10  # Fixed cost per transshipment (either direction)

        # Returning as a dictionary
        return {
            'L': L,
            'LT': LT,
            'demand_level': demand_level,
            'epi_len': epi_len,
            'num_retailer': num_retailer,
            'ini_inv': ini_inv,
            'holding': holding,
            'lost_sales': lost_sales,
            'capacity': capacity,
            'fixed_order': fixed_order,
            'per_trans_item': per_trans_item,
            'per_trans_order': per_trans_order
        }

if __name__ == '__main__':
    scenario = "sN3h_1_5_10b2"
    scenarioDesign = ScenarioDesign(scenario)
    parameters = scenarioDesign.get_parameter()
    print(parameters)

    scenario = "sN2h_1_5b2"
    scenarioDesign = ScenarioDesign(scenario)
    parameters = scenarioDesign.get_parameter()
    print(parameters)

    scenario = "mN3h_10_50_50b3"
    scenarioDesign = ScenarioDesign(scenario)
    parameters = scenarioDesign.get_parameter()
    print(parameters)