import sys
import re
import pandas as pd
import numpy as np

sys.path

class ScenarioDesign_rental():
    def __init__(self, scenario):
        """
        Initialize the PCDiversityCalculator with a population of individuals.

        :param population: List of individuals, where each individual is a list of PC values.
        """
        self.scenario = scenario

    def get_parameter(self, seed=0):

        #the following is for PSO, must have [0,0,0], represents do not rental
        rental_choice = [[0, 0, 0],
                         [40, 100, 1], [160, 500, 1], [200, 700, 1],
                         [20, 100, 6], [80, 500, 6], [100, 700, 6]]

        # the following is for GP
        # rental_choice = [[40, 100, 1], [160, 500, 1], [200, 700, 1],
        #                 [20, 100, 6], [80, 500, 6], [100, 700, 6]]

        if self.scenario == "teckwah_training":
            # Calculated values
            L = 2  # Length of forecast horizon (default)
            LT = 2
            epi_len = 64  # Length of one episode (default)
            demand_level = 30000
            num_retailer = 2
            ini_inv = [500] * 2  # Initial inventory levels
            holding = [2, 10]  # Holding costs
            lost_sales = [50, 100]  # Per unit lost sales costs
            capacity = [50000, 50000]  # Inventory capacities
            production_capacity = [500, 500]  # Production capacities
            fixed_order = [1000, 800]  # Fixed order costs per order
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
                'production_capacity': production_capacity,
                'fixed_order': fixed_order,
                'per_trans_item': per_trans_item,
                'per_trans_order': per_trans_order,
                'rental_choice': rental_choice
            }
        elif self.scenario == "teckwah_test":
            # Obtain testing demand data from csv file
            test_demand = pd.read_csv('./Utils/teckwah.csv')
            # test_demand = pd.read_csv('teckwah.csv')
            demand_test = []
            np.random.seed(seed)
            # for i in range(10): # get 10 instances
            # demand_hist_list = test_demand.iloc[2 * k: 2 * k + 2, 1:].to_numpy()
            k_1 = np.random.randint(7)
            k_2 = np.random.randint(7)
            # print("K_1" + str(k_1))
            # print("K_2" + str(k_2))
            demand_hist_list_site1 = test_demand.iloc[k_1, 1:].to_numpy()
            demand_hist_list_site2 = test_demand.iloc[k_2, 1:].to_numpy()
            demand_hist_list = np.array([demand_hist_list_site1, demand_hist_list_site2])
                # demand_test.append(demand_hist_list)
            # Calculated values
            L = 2  # Length of forecast horizon (default)
            LT = 2
            demand_level = None
            epi_len = 64  # Length of one episode (default)
            num_retailer = 2
            ini_inv = [500] * 2  # Initial inventory levels
            holding = [2, 10]  # Holding costs
            lost_sales = [50, 100]  # Per unit lost sales costs
            capacity = [50000, 50000]  # Inventory capacities
            production_capacity = [500, 500]  # Production capacities
            fixed_order = [1000, 800]  # Fixed order costs per order
            per_trans_item = 1  # Per unit cost for transshipment (either direction)
            per_trans_order = 10  # Fixed cost per transshipment (either direction)
            # Returning as a dictionary
            return {
                'L': L,
                'LT': LT,
                'demand_level': demand_level,
                'demand_test': demand_hist_list,
                'epi_len': epi_len,
                'num_retailer': num_retailer,
                'ini_inv': ini_inv,
                'holding': holding,
                'lost_sales': lost_sales,
                'capacity': capacity,
                'production_capacity': production_capacity,
                'fixed_order': fixed_order,
                'per_trans_item': per_trans_item,
                'per_trans_order': per_trans_order,
                'rental_choice': rental_choice
            }
        else:
            # Assuming the string format is: "s/m/lN<retailers>h_<holding1>_<holding2>_<holding3>b<LT>"
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
            production_capacity = [c / 50 for c in capacity]  # Production capacities
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
                'production_capacity': production_capacity,
                'fixed_order': fixed_order,
                'per_trans_item': per_trans_item,
                'per_trans_order': per_trans_order,
                'rental_choice': rental_choice
            }

    def get_parameter_S2Demo(self, seed=0, epi_len = 64):

        #the following is for PSO, must have [0,0,0], represents do not rental
        rental_choice = [[0, 0, 0],
                         [40, 100, 1], [160, 500, 1], [200, 700, 1],
                         [20, 100, 6], [80, 500, 6], [100, 700, 6]]

        # the following is for GP
        # rental_choice = [[40, 100, 1], [160, 500, 1], [200, 700, 1],
        #                 [20, 100, 6], [80, 500, 6], [100, 700, 6]]

        if self.scenario == "teckwah_training":
            # Calculated values
            L = 2  # Length of forecast horizon (default)
            LT = 2
            # epi_len = 64  # Length of one episode (default)
            demand_level = 30000
            num_retailer = 2
            ini_inv = [500] * 2  # Initial inventory levels
            holding = [2, 10]  # Holding costs
            lost_sales = [50, 100]  # Per unit lost sales costs
            capacity = [50000, 50000]  # Inventory capacities
            production_capacity = [500, 500]  # Production capacities
            fixed_order = [1000, 800]  # Fixed order costs per order
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
                'production_capacity': production_capacity,
                'fixed_order': fixed_order,
                'per_trans_item': per_trans_item,
                'per_trans_order': per_trans_order,
                'rental_choice': rental_choice
            }
        elif self.scenario == "teckwah_test":
            # Obtain testing demand data from csv file
            test_demand = pd.read_csv('./Utils/teckwah.csv')
            # test_demand = pd.read_csv('teckwah.csv')
            demand_test = []
            np.random.seed(seed)
            # for i in range(10): # get 10 instances
            # demand_hist_list = test_demand.iloc[2 * k: 2 * k + 2, 1:].to_numpy()
            k_1 = np.random.randint(7)
            k_2 = np.random.randint(7)
            # print("K_1" + str(k_1))
            # print("K_2" + str(k_2))
            demand_hist_list_site1 = test_demand.iloc[k_1, 1:].to_numpy()
            demand_hist_list_site2 = test_demand.iloc[k_2, 1:].to_numpy()
            demand_hist_list = np.array([demand_hist_list_site1, demand_hist_list_site2])
                # demand_test.append(demand_hist_list)
            # Calculated values
            L = 2  # Length of forecast horizon (default)
            LT = 2
            demand_level = None
            # epi_len = 64  # Length of one episode (default)
            num_retailer = 2
            ini_inv = [500] * 2  # Initial inventory levels
            holding = [2, 10]  # Holding costs
            lost_sales = [50, 100]  # Per unit lost sales costs
            capacity = [50000, 50000]  # Inventory capacities
            production_capacity = [500, 500]  # Production capacities
            fixed_order = [1000, 800]  # Fixed order costs per order
            per_trans_item = 1  # Per unit cost for transshipment (either direction)
            per_trans_order = 10  # Fixed cost per transshipment (either direction)
            # Returning as a dictionary
            return {
                'L': L,
                'LT': LT,
                'demand_level': demand_level,
                'demand_test': demand_hist_list,
                'epi_len': epi_len,
                'num_retailer': num_retailer,
                'ini_inv': ini_inv,
                'holding': holding,
                'lost_sales': lost_sales,
                'capacity': capacity,
                'production_capacity': production_capacity,
                'fixed_order': fixed_order,
                'per_trans_item': per_trans_item,
                'per_trans_order': per_trans_order,
                'rental_choice': rental_choice
            }
        else:
            # Assuming the string format is: "s/m/lN<retailers>h_<holding1>_<holding2>_<holding3>b<LT>"
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
            # epi_len = 64  # Length of one episode (default)
            ini_inv = [10] * num_retailer  # Initial inventory levels
            lost_sales = [b * h for h in holding]  # Per unit lost sales costs
            capacity = [5 * demand_level] * num_retailer  # Inventory capacities
            production_capacity = [c / 50 for c in capacity]  # Production capacities
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
                'production_capacity': production_capacity,
                'fixed_order': fixed_order,
                'per_trans_item': per_trans_item,
                'per_trans_order': per_trans_order,
                'rental_choice': rental_choice
            }



if __name__ == '__main__':
    # scenario = "sN3h_1_5_10b2"
    # scenarioDesign = ScenarioDesign(scenario)
    # parameters = scenarioDesign.get_parameter()
    # print(parameters)
    #
    # scenario = "sN2h_1_5b2"
    # scenarioDesign = ScenarioDesign(scenario)
    # parameters = scenarioDesign.get_parameter()
    # print(parameters)
    #
    # scenario = "mN3h_10_50_50b3"
    # scenarioDesign = ScenarioDesign(scenario)
    # parameters = scenarioDesign.get_parameter()
    # print(parameters)

    scenario = "teckwah_training"
    scenarioDesign = ScenarioDesign_rental(scenario)
    parameters = scenarioDesign.get_parameter()
    print(parameters)

    scenario = "teckwah_test"
    scenarioDesign = ScenarioDesign_rental(scenario)
    parameters = scenarioDesign.get_parameter()
    print(parameters)

    scenario = "teckwah_test"
    scenarioDesign = ScenarioDesign_rental(scenario)
    parameters = scenarioDesign.get_parameter(seed=6)
    print(parameters)