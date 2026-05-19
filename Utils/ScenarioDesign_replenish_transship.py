import re

import numpy as np
import pandas as pd


class ScenarioDesign_replenish_transship():
    def __init__(self, scenario):
        self.scenario = scenario

    def get_parameter(self, seed=0):
        return self._build_parameters(seed, epi_len=64)

    def get_parameter_S2Demo(self, seed=0, epi_len=64):
        return self._build_parameters(seed, epi_len=epi_len)

    def _build_parameters(self, seed, epi_len):
        if self.scenario == "teckwah_training":
            return self._base_parameters(
                demand_level=30000,
                epi_len=epi_len,
                num_retailer=2,
                ini_inv=[500] * 2,
                holding=[2, 10],
                lost_sales=[50, 100],
                capacity=[50000, 50000],
                production_capacity=[500, 500],
                fixed_order=[1000, 800],
                per_unit_order=[1, 1],
            )

        if self.scenario == "teckwah_test":
            test_demand = pd.read_csv('./Utils/teckwah.csv')
            np.random.seed(seed)
            k_1 = np.random.randint(7)
            k_2 = np.random.randint(7)
            demand_hist_list_site1 = test_demand.iloc[k_1, 1:].to_numpy()
            demand_hist_list_site2 = test_demand.iloc[k_2, 1:].to_numpy()
            demand_hist_list = np.array([demand_hist_list_site1, demand_hist_list_site2])

            parameters = self._base_parameters(
                demand_level=30000,
                epi_len=epi_len,
                num_retailer=2,
                ini_inv=[500] * 2,
                holding=[2, 10],
                lost_sales=[50, 100],
                capacity=[50000, 50000],
                production_capacity=[500, 500],
                fixed_order=[1000, 800],
                per_unit_order=[1, 1],
            )
            parameters['demand_test'] = demand_hist_list
            return parameters

        parts = re.split('[Nhb]', self.scenario)
        demand_scale = parts[0]
        if demand_scale == "s":
            demand_level = 20
        elif demand_scale == "m":
            demand_level = 100
        else:
            demand_level = 1000

        num_retailer = int(parts[1])
        holding = [int(item) for item in parts[2].split('_')[1:]]
        lead_time_factor = int(parts[3])
        capacity = [5 * demand_level] * num_retailer

        return self._base_parameters(
            demand_level=demand_level,
            epi_len=epi_len,
            num_retailer=num_retailer,
            ini_inv=[10] * num_retailer,
            holding=holding,
            lost_sales=[lead_time_factor * item for item in holding],
            capacity=capacity,
            production_capacity=[item / 5 for item in capacity],
            fixed_order=[20] * num_retailer,
            per_unit_order=[1] * num_retailer,
        )

    def _base_parameters(self, demand_level, epi_len, num_retailer, ini_inv, holding, lost_sales,
                         capacity, production_capacity, fixed_order, per_unit_order):
        return {
            'L': 2,
            'LT': 2,
            'demand_level': demand_level,
            'epi_len': epi_len,
            'num_retailer': num_retailer,
            'ini_inv': ini_inv,
            'holding': holding,
            'lost_sales': lost_sales,
            'capacity': capacity,
            'production_capacity': production_capacity,
            'fixed_order': fixed_order,
            'per_unit_order': per_unit_order,
            'per_trans_item': 1,
            'per_trans_order': 10,
        }


if __name__ == '__main__':
    scenario_design = ScenarioDesign_replenish_transship("sN2h_1_5b2")
    print(scenario_design.get_parameter())