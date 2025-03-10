import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import pandas as pd
import pytz
import math
from datetime import datetime
from torch.distributions import Categorical

from MTGP_niching_rental_RFQ.replenishment import *
from MTGP_niching_rental_RFQ.transshipment import *
from MTGP_niching_rental_RFQ.rental import *
from MTGP_niching_rental_RFQ.RFQ_predict import *
import threading
import MTGP_niching_rental_RFQ.logistic_util as logistic_util

# np.random.seed(0)
# ###############training##########################################
# # Parameters
# L = 2 # Length of forecast horizon
# LT = 2 # Lead time
# demand_level = 20
# epi_len = 64 # Length of one episode
# num_retailer = 3 # Number of sites/retailers
# ini_inv = [10,10,10] # Initial inventory levels
# holding = [1, 5, 10] # Holding costs
# lost_sales = 2 * holding # Per unit lost sales costs
# capacity = [5*demand_level,5*demand_level,5*demand_level] # Inventory capacities
# fixed_order = [10,10,10] # Fixed order costs per order
# per_trans_item = 5 # Per unit cost for transshipment (either direction)
# per_trans_order = 20 # Fixed cost per transshipment (either direction)
# #########################################################


# Demand forecast function
# Note that at decision time t, demand for time t has already been realised
class SupplierSupport:
    def __init__(self, seed, support_level, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.support_level = support_level
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.list = np.random.uniform(0, self.support_level, size=(self.num_retailer, self.epi_len + 3)) # for Teckwah
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(self.list[1])):
        #     self.list[1][i] = 0

    def seedRotation(self): # add by xumeng for changing to a new seed
        self.seed = self.seed + 1000
        np.random.seed(self.seed)
    def reset(self):
        self.seedRotation() # add by xumeng for changing to a new seed
        self.list = np.random.uniform(0, self.support_level, size=(self.num_retailer, self.epi_len + 3))# for Teckwah
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(self.list[1])):
        #     self.list[1][i] = 0

    def f(self, n, t):  # Generate forecasts, f(n,t) corresponds to demand mean for retailer n at time t+1
        if n >= self.num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    # Function to generate demand history for the two retailers, of length epi_len+1
    def gen_support(self):
        support_hist_list = []  # List to hold demand histories for multiple retailers
        for k in range(self.num_retailer):
            support_hist = []
            for i in range(1, self.epi_len + 2):  # 1 extra demand generated so that last state has a next state
                random_support = np.random.poisson(self.list[k, i])  # Poisson distribution with forecasted mean
                support_hist.append(random_support)
            support_hist_list.append(support_hist)
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(demand_hist_list[1])):
        #     demand_hist_list[1][i] = 0
        return support_hist_list


# Demand forecast function
# Note that at decision time t, demand for time t has already been realised
class RandomDemand:
    def __init__(self, seed, demand_level, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.demand_level = demand_level
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.list = np.random.uniform(0, self.demand_level, size=(self.num_retailer, self.epi_len + 3)) # for Teckwah
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(self.list[1])):
        #     self.list[1][i] = 0

    def seedRotation(self): # add by xumeng for changing to a new seed
        self.seed = self.seed + 1000
        np.random.seed(self.seed)
    def reset(self):
        self.seedRotation() # add by xumeng for changing to a new seed
        self.list = np.random.uniform(0, self.demand_level, size=(self.num_retailer, self.epi_len + 3))# for Teckwah
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(self.list[1])):
        #     self.list[1][i] = 0

    def f(self, n, t):  # Generate forecasts, f(n,t) corresponds to demand mean for retailer n at time t+1
        if n >= self.num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    # Function to generate demand history for the two retailers, of length epi_len+1
    def gen_demand(self):
        demand_hist_list = []  # List to hold demand histories for multiple retailers
        for k in range(self.num_retailer):
            demand_hist = []
            for i in range(1, self.epi_len + 2):  # 1 extra demand generated so that last state has a next state
                random_demand = np.random.poisson(self.list[k, i])  # Poisson distribution with forecasted mean
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        # todo: the following is modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(demand_hist_list[1])):
        #     demand_hist_list[1][i] = 0
        return demand_hist_list

    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []  # List to hold demand histories for multiple retailers
        for k in range(self.num_retailer):
            demand_hist = []
            for i in range(1, self.epi_len + 2):  # 1 extra demand generated so that last state has a next state
                random_demand = np.random.uniform(0,self.list[k, i])  # Poisson distribution with forecasted mean
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0 # urgent RFQ not always happen! by mengxu 2025.3.3
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        # todo: the following is modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(demand_hist_list[1])):
        #     demand_hist_list[1][i] = 0
        return demand_hist_list

class TeckwahDemand:
    def __init__(self, seed, demand_hist_list, forcast, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.demand_hist_list = demand_hist_list
        self.list = forcast
        # for i in range(len(self.list[1])):
        #     self.list[1][i] = 0

    def seedRotation(self): # add by xumeng for changing to a new seed
        self.seed = self.seed + 1000
        np.random.seed(self.seed)
    def reset(self):
        self.seedRotation() # add by xumeng for changing to a new seed

    def f(self, n, t):  # Generate forecasts, f(n,t) corresponds to demand mean for retailer n at time t+1
        if n >= self.num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    # Function to generate demand history for the two retailers, of length epi_len+1
    def gen_demand(self):
        return self.demand_hist_list

    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []  # List to hold demand histories for multiple retailers
        for k in range(self.num_retailer):
            demand_hist = []
            for i in range(1, self.epi_len + 2):  # 1 extra demand generated so that last state has a next state
                random_demand = np.random.random(self.list[k, i])  # Poisson distribution with forecasted mean
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0 # urgent RFQ not always happen! by mengxu 2025.3.3
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(demand_hist_list[1])):
        #     demand_hist_list[1][i] = 0
        return demand_hist_list


class Retailer:
    def __init__(self, demand_records, number, f,
                 ini_inv, holding, lost_sales, L, LT, capacity, production_capacity, fixed_order,
                 per_trans_item, per_trans_order, support_records, support_unit_cost, RFQ_happen_pro = -1.0):  # Each retailer has its own number 0,1,2,... f is its forecast
        self.ini_inv = ini_inv
        self.L = L
        self.LT = LT
        self.number = number  # Retailer number
        self.inv_level = ini_inv[number]  # Initial inventory level
        self.holding_cost = holding[number]  # Holding cost per unit
        self.lost_sales_cost = lost_sales[number]  # Cost per unit of lost sales
        self.pipeline = [0] * (LT - 1)  # List to track orders yet to arrive, of length (LT - 1)
        self.forecast = [f(number, t) for t in range(1, L + 1)]  # Forecast for time t+1
        self.capacity = capacity[number]  # Maximum inventory capacity
        self.production_capacity = production_capacity[number]
        self.demand_list = demand_records  # Historical demand records
        self.fixed_order_cost = fixed_order[number]  # Fixed cost for placing an order
        self.transshipment_cost = per_trans_item
        self.fixed_order_transshipment_cost = per_trans_order
        self.action = 0  # Order qty
        self.support_list = support_records # Historical support records
        self.RFQ_happen_pro = RFQ_happen_pro
        self.support_unit_cost = support_unit_cost

    def reset(self, f):
        self.inv_level = self.ini_inv[self.number]
        self.pipeline = [0] * (self.LT - 1)
        self.forecast = [f(self.number, t) for t in range(1, self.L + 1)]  # Forecast for time t+1

    def order_arrival(self, demand, rental_available):  # Get next state after pipeline inv arrives and demand is realized
        self.inv_level = min(self.capacity,
                             self.inv_level + self.pipeline[0])  # Pipeline arrives, cannot exceed storage capacity
        self.inv_level -= demand

        if self.inv_level < 0:
            if np.absolute(self.inv_level) < rental_available:
                self.inv_level = 0
                rental_available = rental_available - np.absolute(self.inv_level)
            else:
                self.inv_level = self.inv_level + rental_available
                rental_available = 0
        if rental_available < 0:
            print("Error! rental_available should not be smaller than 0!!!")
        # Update pipeline
        self.pipeline = np.concatenate((self.pipeline[1:], [self.action]))
        return rental_available

    def urgent_RFQ_order_arrival(self, RFQ_demand, rental_available, predict_support_inventory, true_support_inventory):  # Get next state after pipeline inv arrives and demand is realized
        self.inv_level = min(self.capacity,
                             self.inv_level)  # Pipeline arrives, cannot exceed storage capacity
        self.inv_level -= RFQ_demand

        support_cost = 0

        if self.inv_level < 0:
            if np.absolute(self.inv_level) < rental_available:
                self.inv_level = 0
                rental_available = rental_available - np.absolute(self.inv_level)
            else:
                self.inv_level = self.inv_level + rental_available
                rental_available = 0
                #todo: need to double-check this part!
                if predict_support_inventory <= true_support_inventory:
                    if np.absolute(self.inv_level) < predict_support_inventory:
                        support_cost = np.absolute(self.inv_level) * self.support_unit_cost
                    elif np.absolute(self.inv_level) > predict_support_inventory:
                        support_cost = predict_support_inventory * self.support_unit_cost + (np.absolute(self.inv_level)-predict_support_inventory) * self.lost_sales_cost
                    self.inv_level = 0
                else:
                    if np.absolute(self.inv_level) < true_support_inventory:
                        support_cost = np.absolute(self.inv_level) * self.support_unit_cost
                    elif np.absolute(self.inv_level) > true_support_inventory:
                        support_cost = true_support_inventory * self.support_unit_cost + (np.absolute(self.inv_level)-true_support_inventory)* self.lost_sales_cost
                    self.inv_level = 0
        if rental_available < 0:
            print("Error! rental_available should not be smaller than 0!!!")
        # Update pipeline
        # self.pipeline = np.concatenate((self.pipeline[1:], [self.action]))
        return rental_available, support_cost

class TimeoutException(Exception):
    pass

class InvOptEnv:
    def __init__(self, seed, parameters):
        """
        Initialize the inventory optimization environment with the given parameters.

        :param seed: Random seed for reproducibility.
        :param L: Length of forecast horizon.
        :param LT: Lead time.
        :param demand_level: Demand level.
        :param epi_len: Length of one episode.
        :param num_retailer: Number of sites/retailers.
        :param ini_inv: Initial inventory levels.
        :param holding: Holding costs.
        :param fixed_order: Fixed order costs per order.
        :param per_trans_item: Per unit cost for transshipment (either direction).
        :param per_trans_order: Fixed cost per transshipment (either direction).
        """
        # Parameters
        self.L = parameters['L']
        self.LT = parameters['LT']
        self.demand_level = parameters['demand_level']
        self.epi_len = parameters['epi_len']
        self.num_retailer = parameters['num_retailer']
        self.ini_inv = parameters['ini_inv']
        self.holding = parameters['holding']
        self.lost_sales = parameters['lost_sales']
        self.capacity = parameters['capacity']
        self.production_capacity = parameters['production_capacity']
        self.fixed_order = parameters['fixed_order']
        self.per_trans_item = parameters['per_trans_item']
        self.per_trans_order = parameters['per_trans_order']
        self.support_level = parameters['support_level']
        self.support_unit_cost = parameters['support_unit_cost'] #todo: need to find a suitable value
        self.RFQ_happen_pro = parameters['RFQ_happen_pro']
        self.partial_information_visibility = parameters['partial_information_visibility']
        # add by xu meng 2024.12.2
        self.rental_choice = parameters['rental_choice']
        self.current_rentals = []

        if self.demand_level == None:#use teckwah dataset
            self.demand_records = parameters['demand_test']
            # Update forecasts
            forecast1_all = []
            forecast2_all = []
            for current_period in range(len(self.demand_records[0])):
                forecast1 = [self.demand_records[0, current_period]]
                forecast2 = [self.demand_records[1, current_period]]
                # forecast1 = [self.demand_records[0, k] for k in range(current_period, current_period + self.L)]
                # forecast2 = [self.demand_records[1, k] for k in range(current_period, current_period + self.L)]
                forecast1_all = forecast1_all + forecast1
                forecast2_all = forecast2_all + forecast2
            forecast = np.array([forecast1_all, forecast2_all])
            self.rd = TeckwahDemand(seed, self.demand_records, forecast, self.num_retailer, self.epi_len)
            self.urgent_RFQ_demand_records = self.rd.gen_urgent_RFQ_demand(
                self.RFQ_happen_pro)  # add by meng xu for urgent RFQ
        else:
            self.rd = RandomDemand(seed, self.demand_level, self.num_retailer, self.epi_len)
            self.demand_records = self.rd.gen_demand()
            self.urgent_RFQ_demand_records = self.rd.gen_urgent_RFQ_demand(self.RFQ_happen_pro) # add by meng xu for urgent RFQ

        # this is for RFQ and support for partial information visibility
        supplierSupport = SupplierSupport(seed, self.support_level, self.num_retailer, self.epi_len)
        self.support_records = supplierSupport.gen_support()

        self.n_retailers = self.num_retailer
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(self.demand_records[i], i, self.rd.f,
                                           self.ini_inv, self.holding, self.lost_sales,
                                           self.L, self.LT, self.capacity, self.production_capacity, self.fixed_order,
                                           self.per_trans_item, self.per_trans_order,
                                           self.support_records, self.support_unit_cost, self.RFQ_happen_pro))

        self.n_period = len(self.demand_records[0])
        self.current_period = 1
        self.state = []  # include replenishment state of each retailer and transshipment state of each pair of sites
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment_retailer = np.array([
                retailer.inv_level, retailer.holding_cost,
                retailer.lost_sales_cost, retailer.capacity,
                retailer.fixed_order_cost, retailer.pipeline[0],  # only suitable for LT = 2
                retailer.forecast[0], retailer.forecast[1],
                retailer.transshipment_cost, retailer.fixed_order_transshipment_cost
            ])  # only suitable for LT = 2
            state_replenishment.append(state_replenishment_retailer)
        self.state.append(state_replenishment)

        state_transshipment = []
        for i in range(len(self.retailers)):
            retailer_i = self.retailers[i]
            for j in range(i + 1, len(self.retailers)):
                retailer_j = self.retailers[j]
                state_transshipment_retailer_pair = np.array([
                    i, j,  # store the id, not used for decision, but for know which pair
                    retailer_i.inv_level, retailer_i.holding_cost,
                    retailer_i.lost_sales_cost, retailer_i.capacity,
                    retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                    # only suitable for LT = 2
                    retailer_i.forecast[0], retailer_i.forecast[1],
                    retailer_j.inv_level, retailer_j.holding_cost,
                    retailer_j.lost_sales_cost, retailer_j.capacity,
                    retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                    # only suitable for LT = 2
                    retailer_j.forecast[0], retailer_j.forecast[1],
                    retailer_i.transshipment_cost, retailer_i.fixed_order_transshipment_cost
                ])
                state_transshipment.append(state_transshipment_retailer_pair)
        self.state.append(state_transshipment)

        state_rental = []
        total_current_rental = 0
        if len(self.current_rentals) != 0:
            total_current_rental = sum(each_current_rental[1] for each_current_rental in self.current_rentals)
            # Filter and update the current rentals, mainly for rental length
            self.current_rentals = [
                [each[0], each[1], each[2] - 1] if each[2] > 1 else each
                for each in self.current_rentals
                if each[2] != 1
            ]
        for each_rental_choice in self.rental_choice:
            each_rental_state = [total_current_rental, each_rental_choice[0], each_rental_choice[1],
                                 each_rental_choice[2]]
            state_rental.append(each_rental_state)
        self.state.append(state_rental)

    def timeout_handler(self):
        raise TimeoutException("Operation timed out!")

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        """
        Runs a function with a specified timeout. If the function takes longer than
        `timeout` seconds, it will return `np.inf` instead.

        Parameters:
        - func: The function to run.
        - timeout: The maximum time in seconds the function is allowed to run.
        - args, kwargs: Arguments to pass to the function.
        """
        timer = threading.Timer(timeout, self.timeout_handler)
        timer.start()
        try:
            result = func(*args, **kwargs)
        except TimeoutException:
            print("Function took too long to run!")
            result = np.nan
        finally:
            timer.cancel()  # Cancel the timer if the function completes within the timeout
        return result

    def reset(self):  # Resets state of all retailers and DCs by calling their respective reset methods
        self.rd.reset()
        self.demand_records = self.rd.gen_demand()  # New instance of demand realizations
        for retailer in self.retailers:
            retailer.reset(self.rd.f)
        self.current_period = 1
        self.state = []  # include replenishment state of each retailer and transshipment state of each pair of sites
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment_retailer = np.array([retailer.inv_level, retailer.holding_cost,
                                                     retailer.lost_sales_cost, retailer.capacity,
                                                     retailer.production_capacity,
                                                     retailer.fixed_order_cost, retailer.pipeline[0],
                                                     # only suitable for LT = 2
                                                     retailer.forecast[0], retailer.forecast[1],
                                                     retailer.transshipment_cost,
                                                     retailer.fixed_order_transshipment_cost])  # only suitable for LT = 2
            state_replenishment.append(state_replenishment_retailer)
        self.state.append(state_replenishment)
        state_transshipment = []
        for i in range(len(self.retailers)):
            retailer_i = self.retailers[i]
            for j in range(i + 1, len(self.retailers)):
                retailer_j = self.retailers[j]
                state_transshipment_retailer_pair = np.array(
                    [i, j,  # store the id, not used for decision, but for know which pair
                     retailer_i.inv_level, retailer_i.holding_cost,
                     retailer_i.lost_sales_cost, retailer_i.capacity,
                     retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                     # only suitable for LT = 2
                     retailer_i.forecast[0], retailer_i.forecast[1],
                     retailer_j.inv_level, retailer_j.holding_cost,
                     retailer_j.lost_sales_cost, retailer_j.capacity,
                     retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                     # only suitable for LT = 2
                     retailer_j.forecast[0], retailer_j.forecast[1],
                     retailer_j.transshipment_cost, retailer_j.fixed_order_transshipment_cost])
                state_transshipment.append(state_transshipment_retailer_pair)
        self.state.append(state_transshipment)

        state_rental = []
        total_current_rental = 0
        if len(self.current_rentals) != 0:
            total_current_rental = sum(each_current_rental[1] for each_current_rental in self.current_rentals)
            # Filter and update the current rentals, mainly for rental length
            self.current_rentals = [
                [each[0], each[1], each[2] - 1] if each[2] > 1 else each
                for each in self.current_rentals
                if each[2] != 1
            ]
        for each_rental_choice in self.rental_choice:
            each_rental_state = [total_current_rental, each_rental_choice[0], each_rental_choice[1],
                                 each_rental_choice[2]]
            state_rental.append(each_rental_state)
        self.state.append(state_rental)

        state_RFQ_predict= []
        for retailer_index in range(len(self.retailers)):
            retailer = self.retailers[retailer_index]
            state_RFQ_predict_retailer = np.array([self.urgent_RFQ_demand_records[retailer_index][self.current_period-2], retailer.inv_level])  # only suitable for LT = 2
            state_RFQ_predict.append(state_RFQ_predict_retailer)
        self.state.append(state_RFQ_predict)
        return self.state

    def step_value(self, action_modified):  # modified by mengxu to make it not only suitable for 2 sites
        if len(self.retailers) == 2:
            all_cost = []

            # Update inv levels and pipelines
            total_current_rental = 0
            if len(self.current_rentals) != 0:
                total_current_rental = sum(each_current_rental[1] for each_current_rental in self.current_rentals)
                # Filter and update the current rentals, mainly for rental length
                self.current_rentals = [
                    [each[0], each[1], each[2] - 1] if each[2] > 1 else each
                    for each in self.current_rentals
                    if each[2] != 1
                ]
            rental_available = total_current_rental
            for retailer, demand in zip(self.retailers, self.demand_records):
                rental_available = retailer.order_arrival(demand[self.current_period - 2], rental_available)  # -2 not -1

            # this is for handle urgent RFQ demand
            total_support_cost = 0
            RFQ_predict_decisions = action_modified[-1]
            for RFQ_predict_decision, retailer, urgent_RFQ_demand, true_support_inventory in zip(
                    RFQ_predict_decisions, self.retailers, self.urgent_RFQ_demand_records,
                    self.support_records):
                if urgent_RFQ_demand[self.current_period - 2] > 0:  # urgent_RFQ happen
                    # true_support_inventory represents the true maximal inventory support by others
                    # predict_support_inventory = 10 # this represents the predict maximal inventory support by others
                    # todo: here need to predict how much our partner/competitive can support by mengxu 2025.3.3
                    if self.partial_information_visibility:
                        predict_support_inventory = RFQ_predict_decision
                        # print("RFQ_predict_decision: ", RFQ_predict_decision)
                    else:
                        predict_support_inventory = true_support_inventory[self.current_period - 2]
                    rental_available, support_cost = retailer.urgent_RFQ_order_arrival(
                        urgent_RFQ_demand[self.current_period - 2], rental_available, predict_support_inventory,
                        true_support_inventory[self.current_period - 2])  # -2 not -1
                    total_support_cost += support_cost

            # Update rental decision and calculate rental cost, new for Strategy version 2.0
            rental_cost = 0
            rental_decisions = action_modified[-2]  # currently each time only rental one choice
            for each_rental_decision in rental_decisions:
                rental_decision = self.rental_choice[each_rental_decision]
                self.current_rentals.append(rental_decision)
                rental_cost = rental_cost + rental_decision[0]

            # # Update rental decision and calculate rental cost, old for Strategy version 1.0
            # rental_decision = self.rental_choice[action_modified[-1]]  # currently each time only rental one choice
            # self.current_rentals.append(rental_decision)
            # rental_cost = rental_decision[0]

            trans = action_modified[0]  # Transshipment quantity, possibly infeasible
            # Make transshipment quantity feasible
            if trans > 0 and self.retailers[0].inv_level < trans:
                trans = 0
            elif trans < 0 and self.retailers[1].inv_level < -trans:
                trans = 0
            trans_cost = np.abs(trans) * self.per_trans_item + (np.abs(trans) != 0) * self.per_trans_order  # Transshipment cost


            hl_cost_total = 0
            order_cost = 0
            # Calculate sum of order, holding, lost sales costs
            for i, retailer in enumerate(self.retailers):  # Iterate through retailers
                retailer.action = action_modified[i + 1]  # Qty ordered by retailer
                if retailer.action > retailer.production_capacity:
                    retailer.action = retailer.production_capacity
                # Get order costs
                order_cost += (retailer.action > 0) * retailer.fixed_order_cost
                # Do transshipment
                if retailer.number == 0:
                    retailer.inv_level -= trans
                else:
                    retailer.inv_level += trans
                # Get holding/lost sales cost
                if retailer.inv_level < 0:  # Get lost sales costs and set to zero
                    hl_cost_total += - retailer.inv_level * retailer.lost_sales_cost
                    retailer.inv_level = 0
                else:
                    hl_cost_total += retailer.inv_level * retailer.holding_cost



            reward = - trans_cost - hl_cost_total - order_cost - rental_cost - total_support_cost
            all_cost.append(trans_cost)
            all_cost.append(hl_cost_total)
            all_cost.append(order_cost)
            all_cost.append(rental_cost)
            all_cost.append(total_support_cost)

            self.current_period += 1
            if self.current_period >= self.n_period:
                terminate = True
            else:
                terminate = False
            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + self.L)]  # No +1
            # # Update inv levels and pipelines
            # for retailer, demand in zip(self.retailers, self.demand_records):
            #     retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
            self.state = []  # include replenishment state of each retailer and transshipment state of each pair of sites
            state_replenishment = []
            for retailer in self.retailers:
                state_replenishment_retailer = np.array([retailer.inv_level, retailer.holding_cost,
                                                         retailer.lost_sales_cost, retailer.capacity,
                                                         retailer.production_capacity,
                                                         retailer.fixed_order_cost, retailer.pipeline[0],
                                                         # only suitable for LT = 2
                                                         retailer.forecast[0], retailer.forecast[1],
                                                         retailer.transshipment_cost,
                                                         retailer.fixed_order_transshipment_cost])  # only suitable for LT = 2
                state_replenishment.append(state_replenishment_retailer)
            self.state.append(state_replenishment)
            state_transshipment = []
            for i in range(len(self.retailers)):
                retailer_i = self.retailers[i]
                for j in range(i + 1, len(self.retailers)):
                    retailer_j = self.retailers[j]
                    state_transshipment_retailer_pair = np.array(
                        [i, j,  # store the id, not used for decision, but for know which pair
                         retailer_i.inv_level, retailer_i.holding_cost,
                         retailer_i.lost_sales_cost, retailer_i.capacity,
                         retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                         # only suitable for LT = 2
                         retailer_i.forecast[0], retailer_i.forecast[1],
                         retailer_j.inv_level, retailer_j.holding_cost,
                         retailer_j.lost_sales_cost, retailer_j.capacity,
                         retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                         # only suitable for LT = 2
                         retailer_j.forecast[0], retailer_j.forecast[1],
                         retailer_j.transshipment_cost, retailer_j.fixed_order_transshipment_cost])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)

            state_rental = []
            total_current_rental = 0
            if len(self.current_rentals) != 0:
                total_current_rental = sum(each_current_rental[1] for each_current_rental in self.current_rentals)
                # Filter and update the current rentals, mainly for rental length
                self.current_rentals = [
                    [each[0], each[1], each[2] - 1] if each[2] > 1 else each
                    for each in self.current_rentals
                    if each[2] != 1
                ]
            for each_rental_choice in self.rental_choice:
                each_rental_state = [total_current_rental, each_rental_choice[0], each_rental_choice[1],
                                     each_rental_choice[2]]
                state_rental.append(each_rental_state)
            self.state.append(state_rental)

            state_RFQ_predict = []
            for retailer_index in range(len(self.retailers)):
                retailer = self.retailers[retailer_index]
                state_RFQ_predict_retailer = np.array(
                    [self.urgent_RFQ_demand_records[retailer_index][self.current_period - 2],
                     retailer.inv_level])  # only suitable for LT = 2
                state_RFQ_predict.append(state_RFQ_predict_retailer)
            self.state.append(state_RFQ_predict)

            return self.state, reward, terminate, all_cost
            # the following is the original
            # self.state = np.array(
            #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
            #                                                            retailer.forecast] + \
            #     [x for retailer in self.retailers for x in retailer.pipeline])
        elif len(self.retailers) == 3:
            all_cost = []

            # this is for handle urgent RFQ demand
            total_support_cost = 0
            RFQ_predict_decisions = action_modified[-1]
            for RFQ_predict_decision, retailer, urgent_RFQ_demand, true_support_inventory in zip(
                    RFQ_predict_decisions, self.retailers, self.urgent_RFQ_demand_records,
                    self.support_records):
                if urgent_RFQ_demand[self.current_period - 2] > 0:  # urgent_RFQ happen
                    # true_support_inventory represents the true maximal inventory support by others
                    # predict_support_inventory = 10 # this represents the predict maximal inventory support by others
                    # todo: here need to predict how much our partner/competitive can support by mengxu 2025.3.3
                    if self.partial_information_visibility:
                        predict_support_inventory = RFQ_predict_decision
                        # print("RFQ_predict_decision: ", RFQ_predict_decision)
                    else:
                        predict_support_inventory = true_support_inventory[self.current_period - 2]
                    rental_available, support_cost = retailer.urgent_RFQ_order_arrival(
                        urgent_RFQ_demand[self.current_period - 2], rental_available, predict_support_inventory,
                        true_support_inventory[self.current_period - 2])  # -2 not -1
                    total_support_cost += support_cost

            # Update inv levels and pipelines
            total_current_rental = 0
            if len(self.current_rentals) != 0:
                total_current_rental = sum(each_current_rental[1] for each_current_rental in self.current_rentals)
                # Filter and update the current rentals, mainly for rental length
                self.current_rentals = [
                    [each[0], each[1], each[2] - 1] if each[2] > 1 else each
                    for each in self.current_rentals
                    if each[2] != 1
                ]
            rental_available = total_current_rental
            for retailer, demand in zip(self.retailers, self.demand_records):
                rental_available = retailer.order_arrival(demand[self.current_period - 2],
                                                          rental_available)  # -2 not -1

            # Update rental decision and calculate rental cost, new for Strategy version 2.0
            rental_cost = 0
            rental_decisions = action_modified[-1]  # currently each time only rental one choice
            for each_rental_decision in rental_decisions:
                rental_decision = self.rental_choice[each_rental_decision]
                self.current_rentals.append(rental_decision)
                rental_cost = rental_cost + rental_decision[0]

            # # Update rental decision and calculate rental cost, old for Strategy version 1.0
            # rental_decision = self.rental_choice[action_modified[-1]]  # currently each time only rental one choice
            # self.current_rentals.append(rental_decision)
            # rental_cost = rental_decision[0]

            trans01 = action_modified[0]  # Transshipment quantity, possibly infeasible
            trans02 = action_modified[1]  # Transshipment quantity, possibly infeasible
            trans12 = action_modified[2]  # Transshipment quantity, possibly infeasible
            # Make transshipment quantity feasible
            if trans01 > 0 and self.retailers[0].inv_level < trans01:
                trans01 = 0
            elif trans01 < 0 and self.retailers[1].inv_level < -trans01:
                trans01 = 0
            trans_cost_01 = np.abs(trans01) * self.per_trans_item + (
                        np.abs(trans01) != 0) * self.per_trans_order  # Transshipment cost
            if trans02 > 0 and self.retailers[0].inv_level - trans01 < trans02:
                trans02 = 0
            elif trans02 < 0 and self.retailers[2].inv_level < -trans02:
                trans02 = 0
            trans_cost_02 = np.abs(trans02) * self.per_trans_item + (
                        np.abs(trans02) != 0) * self.per_trans_order  # Transshipment cost
            if trans12 > 0 and self.retailers[1].inv_level + trans01 < trans12:
                trans12 = 0
            elif trans12 < 0 and self.retailers[2].inv_level + trans02 < -trans12:
                trans12 = 0
            trans_cost_12 = np.abs(trans12) * self.per_trans_item + (
                        np.abs(trans12) != 0) * self.per_trans_order  # Transshipment cost
            trans_cost = trans_cost_01 + trans_cost_02 + trans_cost_12

            hl_cost_total = 0
            order_cost = 0
            # Calculate sum of order, holding, lost sales costs
            for i, retailer in enumerate(self.retailers):  # Iterate through retailers
                retailer.action = action_modified[i + 3]  # Qty ordered by retailer
                # Get order costs
                order_cost += (retailer.action > 0) * retailer.fixed_order_cost
                # Do transshipment
                if retailer.number == 0:
                    retailer.inv_level = retailer.inv_level - trans01 - trans02
                elif retailer.number == 1:
                    retailer.inv_level = retailer.inv_level + trans01 - trans12
                else:
                    retailer.inv_level = retailer.inv_level + trans02 + trans12
                # Get holding/lost sales cost
                if retailer.inv_level < 0:  # Get lost sales costs and set to zero
                    hl_cost_total += - retailer.inv_level * retailer.lost_sales_cost
                    retailer.inv_level = 0
                else:
                    hl_cost_total += retailer.inv_level * retailer.holding_cost

            reward = - trans_cost - hl_cost_total - order_cost - rental_cost - total_support_cost
            all_cost.append(trans_cost)
            all_cost.append(hl_cost_total)
            all_cost.append(order_cost)
            all_cost.append(rental_cost)
            all_cost.append(total_support_cost)

            self.current_period += 1
            if self.current_period >= self.n_period:
                terminate = True
            else:
                terminate = False
            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + self.L)]  # No +1
            # Update inv levels and pipelines
            # for retailer, demand in zip(self.retailers, self.demand_records):
            #     retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
            self.state = []  # include replenishment state of each retailer and transshipment state of each pair of sites
            state_replenishment = []
            for retailer in self.retailers:
                state_replenishment_retailer = np.array([retailer.inv_level, retailer.holding_cost,
                                                         retailer.lost_sales_cost, retailer.capacity,
                                                         retailer.fixed_order_cost, retailer.pipeline[0],
                                                         # only suitable for LT = 2
                                                         retailer.forecast[0], retailer.forecast[1],
                                                         retailer.transshipment_cost,
                                                         retailer.fixed_order_transshipment_cost])  # only suitable for LT = 2
                state_replenishment.append(state_replenishment_retailer)
            self.state.append(state_replenishment)
            state_transshipment = []
            for i in range(len(self.retailers)):
                retailer_i = self.retailers[i]
                for j in range(i + 1, len(self.retailers)):
                    retailer_j = self.retailers[j]
                    state_transshipment_retailer_pair = np.array(
                        [i, j,  # store the id, not used for decision, but for know which pair
                         retailer_i.inv_level, retailer_i.holding_cost,
                         retailer_i.lost_sales_cost, retailer_i.capacity,
                         retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                         # only suitable for LT = 2
                         retailer_i.forecast[0], retailer_i.forecast[1],
                         retailer_j.inv_level, retailer_j.holding_cost,
                         retailer_j.lost_sales_cost, retailer_j.capacity,
                         retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                         # only suitable for LT = 2
                         retailer_j.forecast[0], retailer_j.forecast[1],
                         retailer_j.transshipment_cost, retailer_j.fixed_order_transshipment_cost])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)

            state_rental = []
            total_current_rental = 0
            if len(self.current_rentals) != 0:
                total_current_rental = sum(each_current_rental[1] for each_current_rental in self.current_rentals)
                # Filter and update the current rentals, mainly for rental length
                self.current_rentals = [
                    [each[0], each[1], each[2] - 1] if each[2] > 1 else each
                    for each in self.current_rentals
                    if each[2] != 1
                ]
            for each_rental_choice in self.rental_choice:
                each_rental_state = [total_current_rental, each_rental_choice[0], each_rental_choice[1],
                                     each_rental_choice[2]]
                state_rental.append(each_rental_state)
            self.state.append(state_rental)

            return self.state, reward, terminate, all_cost



    def run(self, individual): # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0
        current_ep_all_cost = [0., 0., 0., 0., 0.]
        current_ep_all_cost = np.array(current_ep_all_cost)

        max_ep_len = self.epi_len  # max timesteps in one episode
        time_step = 0

        sum_outbound = 0 # for test

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            if len(individual) == 1:
                replenishment_policy = individual[0]
            elif len(individual) == 2:
                replenishment_policy = individual[0]
                rental_policy = individual[1]
            elif len(individual) == 3:
                replenishment_policy = individual[0]
                rental_policy = individual[1]
                RFQ_predict_policy = individual[2]

            # ------- strategy 3 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            action_modified = []
            # get transshipment state for all pairs of sites/retailers
            replenishment_state = state[0]
            transshipment_state = state[1]
            rental_state = state[2] # for each choice in rental_state: [current_rental, rental_price, rental_capacity, rental_month, total_rental_requirement]
            RFQ_predict_state = state[3]
            total_rental_requirement = 0
            for each_transshipment_state in transshipment_state:
                transshipment_quantity = 0
                # transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy), 2)
                action_modified.append(transshipment_quantity)

            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_evolve_S(each_replenishment_state, replenishment_policy), 2)
                # print("replenishment_quantity: ", replenishment_quantity)

                # Strategy 2 (sigmoid): constrain the replenishment quantity to [0, production_capacity]
                # Strategy 2: performs better than Strategy 1 based on one run with popsize 200
                production_capacity = each_replenishment_state[4]
                capacity = each_replenishment_state[3]
                upbound_replenishment_quantity = capacity * 3
                if replenishment_quantity > upbound_replenishment_quantity or replenishment_quantity < 0:
                    replenishment_quantity = logistic_util.logistic_scale_and_shift(replenishment_quantity, 0, upbound_replenishment_quantity)
                # print("replenishment_quantity after sigmoid: ", replenishment_quantity)
                if replenishment_quantity > production_capacity:
                    require_quantity = replenishment_quantity - production_capacity
                    total_rental_requirement = total_rental_requirement + require_quantity
                    replenishment_quantity = production_capacity

                # Strategy 1 (original): constrain the replenishment quantity to [0, production_capacity]
                # production_capacity = each_replenishment_state[4]
                # capacity = each_replenishment_state[3]
                # upbound_replenishment_quantity = capacity * 3
                # if replenishment_quantity > upbound_replenishment_quantity:
                #     replenishment_quantity = upbound_replenishment_quantity
                # if replenishment_quantity < 0:
                #     replenishment_quantity = 0
                # # add by xu meng to consider rental
                # if replenishment_quantity > production_capacity:
                #     require_quantity = replenishment_quantity - production_capacity
                #     total_rental_requirement = total_rental_requirement + require_quantity
                #     replenishment_quantity = production_capacity

                #total_rental_requirement = 0 # for testing the effectiveness of sigmoid
                action_modified.append(replenishment_quantity)

            if len(individual) == 1:
                # rental_decision = -1
                rental_decisions = []
                # print("One tree and do not consider rental!")
            else:
                rental_decisions = []
                onlyRentalOne = False
                if onlyRentalOne:
                # Strategy version 1.0: only rental one decision each time, for making rental decision and delete not enough rental choice, by xu meng 2024.12.2
                    current_rental = 0
                    if len(rental_state) > 0:
                        current_rental = rental_state[0][0]
                    if current_rental < total_rental_requirement:
                        all_rental_priority = []
                        for each_rental_state in rental_state:
                            each_rental_state.append(total_rental_requirement)
                            current_rental = each_rental_state[0]
                            rental_capacity = each_rental_state[2]
                            if current_rental+rental_capacity < total_rental_requirement:
                                rental_priority = np.inf
                            else:
                                rental_priority = GP_evolve_rental(each_rental_state, rental_policy)
                            all_rental_priority.append(rental_priority)
                        # Get the index of the minimal value
                        rental_decision = all_rental_priority.index(min(all_rental_priority))
                        rental_decisions.append(rental_decision)
                else:
                    # Strategy version 2.0: only rental n decision each time, by xu meng 2025.1.10
                    all_rental_priority = []
                    for each_rental_state in rental_state:
                        each_rental_state.append(total_rental_requirement)
                        rental_priority = GP_evolve_rental(each_rental_state, rental_policy)
                        all_rental_priority.append(rental_priority)

                    current_rental = 0
                    if len(rental_state) > 0:
                        current_rental = rental_state[0][0]
                    new_current_rental = current_rental
                    try_times = 0
                    while new_current_rental < total_rental_requirement and try_times < 5:
                        try_times = try_times + 1
                        # Get the index of the minimal value
                        rental_decision = all_rental_priority.index(min(all_rental_priority))
                        all_rental_priority[rental_decision] = np.inf
                        rental_decisions.append(rental_decision)

                        for each_rental_decision in rental_decisions:
                            rental_decision_capacity = self.rental_choice[each_rental_decision][1]
                            new_current_rental = new_current_rental + rental_decision_capacity
                    # if len(rental_decisions) > 1:
                    #     print("rental_decisions: ", rental_decisions)
                action_modified.append(rental_decisions)

                RFQ_predict_decisions = []
                upbound_support_quantity = self.support_level * 2

                # Strategy 2: performs better than Strategy 1 based on one run with popsize 200
                for each_RFQ_predict_state in RFQ_predict_state:
                    RFQ_predict_quantity = round(GP_evolve_RFQ_predict(each_RFQ_predict_state, RFQ_predict_policy), 2)
                    # print("RFQ_predict_quantity: ", RFQ_predict_quantity)
                    if RFQ_predict_quantity <= 0 or RFQ_predict_quantity > upbound_support_quantity:
                        RFQ_predict_quantity = logistic_util.logistic_scale_and_shift(RFQ_predict_quantity, 0, upbound_support_quantity)
                    RFQ_predict_decisions.append(RFQ_predict_quantity)


                #Strategy 1
                # for each_RFQ_predict_state in RFQ_predict_state:
                #     RFQ_predict_quantity = round(GP_evolve_RFQ_predict(each_RFQ_predict_state, RFQ_predict_policy), 2)
                #     # print("RFQ_predict_quantity: ", RFQ_predict_quantity)
                #     if RFQ_predict_quantity <= 0:
                #         RFQ_predict_quantity = 0
                #     if RFQ_predict_quantity > upbound_support_quantity:
                #         RFQ_predict_quantity = upbound_support_quantity
                #     RFQ_predict_decisions.append(RFQ_predict_quantity)


                action_modified.append(RFQ_predict_decisions)


            # ------- strategy 3 ---------------------

            # original
            state, reward, done, all_cost = self.step_value(action_modified)

            # todo: to stop bad run and save training time by mengxu 2024.8.27
            # state, reward, done = None, np.nan, False
            # result = self.run_with_timeout(self.step_value, 0.01, action_modified)
            # if result != np.nan:
            #     state, reward, done = result
            # else:
            #     done = True  # Mark the process as done due to timeout

            # print("\nsolution, state, reward: " + str(site1_candidate[index_site1]) + ", " + str(state) + ", " + str(reward))

            time_step += 1
            current_ep_reward += reward
            all_cost = np.array(all_cost)
            current_ep_all_cost += all_cost

            # break; if the episode is over
            if done:
                break

        sum_outbound = sum_outbound / (2*max_ep_len)
        fitness = -current_ep_reward/max_ep_len
        # print("sum_outbound: ", round(sum_outbound,2), "   , fitness: ", round(fitness,2))
        all_cost_final = np.array(current_ep_all_cost)  # Convert list to NumPy array
        all_cost_fit = all_cost_final/max_ep_len
        return fitness, all_cost_fit

    def run_test(self, individual, states=None, actions=None, rewards=None): # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0
        current_ep_all_cost = [0., 0., 0., 0., 0.]
        current_ep_all_cost = np.array(current_ep_all_cost)

        max_ep_len = self.epi_len  # max timesteps in one episode
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            if len(individual) == 1:
                replenishment_policy = individual[0]
            elif len(individual) == 2:
                replenishment_policy = individual[0]
                rental_policy = individual[1]
            elif len(individual) == 3:
                replenishment_policy = individual[0]
                rental_policy = individual[1]
                RFQ_predict_policy = individual[2]

            # ------- strategy 3 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            action_modified = []
            # get transshipment state for all pairs of sites/retailers
            replenishment_state = state[0]
            transshipment_state = state[1]
            rental_state = state[
                2]  # for each choice in rental_state: [current_rental, rental_price, rental_capacity, rental_month, total_rental_requirement]
            RFQ_predict_state = state[3]
            total_rental_requirement = 0
            for each_transshipment_state in transshipment_state:
                transshipment_quantity = 0
                # transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy), 2)
                action_modified.append(transshipment_quantity)

            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_pair_S_test(each_replenishment_state, replenishment_policy), 2)
                # print("replenishment_quantity: ", replenishment_quantity)

                # Strategy 2 (sigmoid): constrain the replenishment quantity to [0, production_capacity]
                # Strategy 2: performs better than Strategy 1 based on one run with popsize 200
                production_capacity = each_replenishment_state[4]
                capacity = each_replenishment_state[3]
                upbound_replenishment_quantity = capacity * 3
                if replenishment_quantity > upbound_replenishment_quantity or replenishment_quantity < 0:
                    replenishment_quantity = logistic_util.logistic_scale_and_shift(replenishment_quantity, 0,
                                                                                    upbound_replenishment_quantity)
                # print("replenishment_quantity after sigmoid: ", replenishment_quantity)
                if replenishment_quantity > production_capacity:
                    require_quantity = replenishment_quantity - production_capacity
                    total_rental_requirement = total_rental_requirement + require_quantity
                    replenishment_quantity = production_capacity

                # Strategy 1 (original): constrain the replenishment quantity to [0, production_capacity]
                # production_capacity = each_replenishment_state[4]
                # capacity = each_replenishment_state[3]
                # upbound_replenishment_quantity = capacity * 3
                # if replenishment_quantity > upbound_replenishment_quantity:
                #     replenishment_quantity = upbound_replenishment_quantity
                # if replenishment_quantity < 0:
                #     replenishment_quantity = 0
                # # add by xu meng to consider rental
                # if replenishment_quantity > production_capacity:
                #     require_quantity = replenishment_quantity - production_capacity
                #     total_rental_requirement = total_rental_requirement + require_quantity
                #     replenishment_quantity = production_capacity

                # total_rental_requirement = 0 # for testing the effectiveness of sigmoid
                action_modified.append(replenishment_quantity)

            if len(individual) == 1:
                # rental_decision = -1
                rental_decisions = []
                # print("One tree and do not consider rental!")
            else:
                rental_decisions = []
                onlyRentalOne = False
                if onlyRentalOne:
                    # Strategy version 1.0: only rental one decision each time, for making rental decision and delete not enough rental choice, by xu meng 2024.12.2
                    current_rental = 0
                    if len(rental_state) > 0:
                        current_rental = rental_state[0][0]
                    if current_rental < total_rental_requirement:
                        all_rental_priority = []
                        for each_rental_state in rental_state:
                            each_rental_state.append(total_rental_requirement)
                            current_rental = each_rental_state[0]
                            rental_capacity = each_rental_state[2]
                            if current_rental + rental_capacity < total_rental_requirement:
                                rental_priority = np.inf
                            else:
                                rental_priority = GP_pair_rental_test(each_rental_state, rental_policy)
                            all_rental_priority.append(rental_priority)
                        # Get the index of the minimal value
                        rental_decision = all_rental_priority.index(min(all_rental_priority))
                        rental_decisions.append(rental_decision)
                else:
                    # Strategy version 2.0: only rental n decision each time, by xu meng 2025.1.10
                    all_rental_priority = []
                    for each_rental_state in rental_state:
                        each_rental_state.append(total_rental_requirement)
                        rental_priority = GP_pair_rental_test(each_rental_state, rental_policy)
                        all_rental_priority.append(rental_priority)

                    current_rental = 0
                    if len(rental_state) > 0:
                        current_rental = rental_state[0][0]
                    new_current_rental = current_rental
                    try_times = 0
                    while new_current_rental < total_rental_requirement and try_times < 5:
                        try_times = try_times + 1
                        # Get the index of the minimal value
                        rental_decision = all_rental_priority.index(min(all_rental_priority))
                        all_rental_priority[rental_decision] = np.inf
                        rental_decisions.append(rental_decision)

                        for each_rental_decision in rental_decisions:
                            rental_decision_capacity = self.rental_choice[each_rental_decision][1]
                            new_current_rental = new_current_rental + rental_decision_capacity
                    # if len(rental_decisions) > 1:
                    #     print("rental_decisions: ", rental_decisions)
            action_modified.append(rental_decisions)

            RFQ_predict_decisions = []
            upbound_support_quantity = self.support_level * 2

            # Strategy 2: performs better than Strategy 1 based on one run with popsize 200
            for each_RFQ_predict_state in RFQ_predict_state:
                RFQ_predict_quantity = round(GP_pair_RFQ_predict_test(each_RFQ_predict_state, RFQ_predict_policy), 2)
                # print("RFQ_predict_quantity: ", RFQ_predict_quantity)
                if RFQ_predict_quantity <= 0 or RFQ_predict_quantity > upbound_support_quantity:
                    RFQ_predict_quantity = logistic_util.logistic_scale_and_shift(RFQ_predict_quantity, 0,
                                                                                  upbound_support_quantity)
                RFQ_predict_decisions.append(RFQ_predict_quantity)

                # Strategy 1
                # for each_RFQ_predict_state in RFQ_predict_state:
                #     RFQ_predict_quantity = round(GP_evolve_RFQ_predict(each_RFQ_predict_state, RFQ_predict_policy), 2)
                #     # print("RFQ_predict_quantity: ", RFQ_predict_quantity)
                #     if RFQ_predict_quantity <= 0:
                #         RFQ_predict_quantity = 0
                #     if RFQ_predict_quantity > upbound_support_quantity:
                #         RFQ_predict_quantity = upbound_support_quantity
                #     RFQ_predict_decisions.append(RFQ_predict_quantity)

            action_modified.append(RFQ_predict_decisions)

            # ------- strategy 3 ---------------------
            if states is not None:
                states.append(state)

            # the original
            state, reward, done, all_cost = self.step_value(action_modified)
            # todo: to stop bad run and save training time by mengxu 2024.8.27
            # state, reward, done = None, np.nan, False
            # result = self.run_with_timeout(self.step_value, 0.01, action_modified)
            # if result != np.nan:
            #     state, reward, done = result
            # else:
            #     done = True  # Mark the process as done due to timeout

            if actions is not None:
                actions.append(action_modified)
            if rewards is not None:
                rewards.append(reward)
            # print("\nsolution, state, reward: " + str(site1_candidate[index_site1]) + ", " + str(state) + ", " + str(reward))

            time_step += 1
            current_ep_reward += reward
            all_cost = np.array(all_cost)
            current_ep_all_cost += all_cost

            # break; if the episode is over
            if done:
                break

        fitness = -current_ep_reward / max_ep_len
        # print("sum_outbound: ", round(sum_outbound,2), "   , fitness: ", round(fitness,2))
        all_cost_final = np.array(current_ep_all_cost)  # Convert list to NumPy array
        all_cost_fit = all_cost_final / max_ep_len
        return fitness, all_cost_fit

