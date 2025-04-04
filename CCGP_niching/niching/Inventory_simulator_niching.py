import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import pytz
import math
from datetime import datetime
from torch.distributions import Categorical

from CCGP_niching.replenishment import *
from CCGP_niching.transshipment import *
import CCGP_niching.niching.ReplenishmentDecisionSituation as ReplenishmentDecisionSituation
import CCGP_niching.niching.TransshipmentDecisionSituation as TransshipmentDecisionSituation

# np.random.seed(0)
###############training##########################################
# Parameters
L = 2 # Length of forecast horizon
LT = 2 # Lead time
demand_level = 20
epi_len = 64 # Length of one episode
num_retailer = 3 # Number of sites/retailers
ini_inv = [10,10,10] # Initial inventory levels
holding = [1, 5, 10] # Holding costs
lost_sales = 20 * holding # Per unit lost sales costs
capacity = [1000,1000,1000] # Inventory capacities
fixed_order = [50,50,50] # Fixed order costs per order
per_trans_item = 2 # Per unit cost for transshipment (either direction)
per_trans_order = 20 # Fixed cost per transshipment (either direction)
#########################################################

# #########################################################
# # Parameters
# L = 2 # Length of forecast horizon
# LT = 2 # Lead time
# epi_len = 256 # Length of one episode
# ini_inv = [25,25] # Initial inventory levels
# holding = [2,2] # Holding costs
# lost_sales = [50,50] # Per unit lost sales costs
# capacity = [60,60] # Inventory capacities
# fixed_order = [10,10] # Fixed order costs per order
# per_trans_item = 0 # Per unit cost for transshipment (either direction)
# per_trans_order = 5 # Fixed cost per transshipment (either direction)
# #########################################################

# Demand forecast function
# Note that at decision time t, demand for time t has already been realised
class RandomDemand:
    def __init__(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.list = np.random.uniform(0, demand_level, size=(num_retailer, epi_len + 3)) # for Teckwah

    def seedRotation(self): # add by xumeng for changing to a new seed
        self.seed = self.seed + 1000
        np.random.seed(self.seed)
    def reset(self):
        self.seedRotation() # add by xumeng for changing to a new seed
        self.list = np.random.uniform(0, demand_level, size=(num_retailer, epi_len + 3))# for Teckwah

    def f(self, n, t):  # Generate forecasts, f(n,t) corresponds to demand mean for retailer n at time t+1
        if n >= num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    # Function to generate demand history for the two retailers, of length epi_len+1
    def gen_demand(self):
        demand_hist_list = []  # List to hold demand histories for multiple retailers
        for k in range(num_retailer):
            demand_hist = []
            for i in range(1, epi_len + 2):  # 1 extra demand generated so that last state has a next state
                random_demand = np.random.poisson(self.list[k, i])  # Poisson distribution with forecasted mean
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        return demand_hist_list



# Define possible actions for inventory management for Teckwah
action_lists = [[0],[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000],[0]]
action_map = [x for x in itertools.product(*action_lists)]

# action_lists = [[0],[0, 5, 10, 15, 20],[0, 5, 10, 15, 20]]
# action_map = [x for x in itertools.product(*action_lists)]


class Retailer:
    def __init__(self, demand_records, number, f):  # Each retailer has its own number 0,1,2,... f is its forecast
        self.number = number  # Retailer number
        self.inv_level = ini_inv[number]  # Initial inventory level
        self.holding_cost = holding[number]  # Holding cost per unit
        self.lost_sales_cost = lost_sales[number]  # Cost per unit of lost sales
        self.pipeline = [0] * (LT - 1)  # List to track orders yet to arrive, of length (LT - 1)
        self.forecast = [f(number, t) for t in range(1, L + 1)]  # Forecast for time t+1
        self.capacity = capacity[number]  # Maximum inventory capacity
        self.demand_list = demand_records  # Historical demand records
        self.fixed_order_cost = fixed_order[number]  # Fixed cost for placing an order
        self.transshipment_cost = per_trans_item
        self.fixed_order_transshipment_cost = per_trans_order
        self.action = 0  # Order qty

    def reset(self, f):
        self.inv_level = ini_inv[self.number]
        self.pipeline = [0] * (LT - 1)
        self.forecast = [f(self.number, t) for t in range(1, L + 1)]  # Forecast for time t+1

    def order_arrival(self, demand):  # Get next state after pipeline inv arrives and demand is realized
        self.inv_level = min(self.capacity,
                             self.inv_level + self.pipeline[0])  # Pipeline arrives, cannot exceed storage capacity
        self.inv_level -= demand
        # Update pipeline
        self.pipeline = np.concatenate((self.pipeline[1:], [self.action]))


class InvOptEnv:
    def __init__(self,seed):
        self.rd = RandomDemand(seed)
        self.demand_records = self.rd.gen_demand()
        self.n_retailers = num_retailer
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(self.demand_records[i], i, self.rd.f))
        self.n_period = len(self.demand_records[0])
        self.current_period = 1
        self.state = [] # include replenishment state of each retailer and transshipment state of each pair of sites
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment_retailer = np.array([retailer.inv_level, retailer.holding_cost,
                                            retailer.lost_sales_cost, retailer.capacity,
                                            retailer.fixed_order_cost, retailer.pipeline[0], #only suitable for LT = 2
                                            retailer.forecast[0], retailer.forecast[1],
                                            retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])    #only suitable for LT = 2
            state_replenishment.append(state_replenishment_retailer)
        self.state.append(state_replenishment)
        state_transshipment = []
        for i in range(len(self.retailers)):
            retailer_i = self.retailers[i]
            for j in range(i+1, len(self.retailers)):
                retailer_j = self.retailers[j]
                state_transshipment_retailer_pair = np.array([i, j, # store the id, not used for decision, but for know which pair
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
                                                    retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])
                state_transshipment.append(state_transshipment_retailer_pair)
        self.state.append(state_transshipment)
        #the following is the original
        # self.state = np.array(
        #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
        #                                                            retailer.forecast] + \
        #     [x for retailer in self.retailers for x in retailer.pipeline])

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
                     retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])
                state_transshipment.append(state_transshipment_retailer_pair)
        self.state.append(state_transshipment)
        #the following is the original
        # self.state = np.array(
        #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
        #                                                            retailer.forecast] + \
        #     [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state

    def step(self, action):  # modified by mengxu to make it not only suitable for 2 sites
        if len(self.retailers) == 2:
            action_modified = action_map[action]
            trans = action_modified[0]  # Transshipment quantity, possibly infeasible
            # Make transshipment quantity feasible
            if trans > 0 and self.retailers[0].inv_level < trans:
                trans = 0
            elif trans < 0 and self.retailers[1].inv_level < -trans:
                trans = 0
            trans_cost = np.abs(trans) * per_trans_item + (np.abs(trans) != 0) * per_trans_order  # Transshipment cost

            hl_cost_total = 0
            order_cost = 0
            # Calculate sum of order, holding, lost sales costs
            for i, retailer in enumerate(self.retailers):  # Iterate through retailers
                retailer.action = action_modified[i + 1]  # Qty ordered by retailer
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
            reward = - trans_cost - hl_cost_total - order_cost

            self.current_period += 1
            if self.current_period >= self.n_period:
                terminate = True
            else:
                terminate = False
            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + L)]  # No +1
            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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
                         retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)
            # the following is the original
            # self.state = np.array(
            #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
            #                                                            retailer.forecast] + \
            #     [x for retailer in self.retailers for x in retailer.pipeline])
        elif len(self.retailers) == 3:
            action_modified = action_map[action]
            trans01 = action_modified[0]  # Transshipment quantity, possibly infeasible
            trans02 = action_modified[1]  # Transshipment quantity, possibly infeasible
            trans12 = action_modified[2]  # Transshipment quantity, possibly infeasible
            # Make transshipment quantity feasible
            if trans01 > 0 and self.retailers[0].inv_level < trans01:
                trans01 = 0
            elif trans01 < 0 and self.retailers[1].inv_level < -trans01:
                trans01 = 0
            trans_cost_01 = np.abs(trans01) * per_trans_item + (
                        np.abs(trans01) != 0) * per_trans_order  # Transshipment cost
            if trans02 > 0 and self.retailers[0].inv_level - trans01 < trans02:
                trans02 = 0
            elif trans02 < 0 and self.retailers[2].inv_level < -trans02:
                trans02 = 0
            trans_cost_02 = np.abs(trans02) * per_trans_item + (
                        np.abs(trans02) != 0) * per_trans_order  # Transshipment cost
            if trans12 > 0 and self.retailers[1].inv_level + trans01 < trans12:
                trans12 = 0
            elif trans12 < 0 and self.retailers[2].inv_level + trans02 < -trans12:
                trans12 = 0
            trans_cost_12 = np.abs(trans12) * per_trans_item + (
                        np.abs(trans12) != 0) * per_trans_order  # Transshipment cost
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
            reward = - trans_cost - hl_cost_total - order_cost

            self.current_period += 1
            if self.current_period >= self.n_period:
                terminate = True
            else:
                terminate = False
            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + L)]  # No +1
            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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
                         retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)

        return self.state, reward, terminate

    def step_value(self, action_modified):  # modified by mengxu to make it not only suitable for 2 sites
        if len(self.retailers) == 2:
            trans = action_modified[0]  # Transshipment quantity, possibly infeasible
            # Make transshipment quantity feasible
            if trans > 0 and self.retailers[0].inv_level < trans:
                trans = 0
            elif trans < 0 and self.retailers[1].inv_level < -trans:
                trans = 0
            trans_cost = np.abs(trans) * per_trans_item + (np.abs(trans) != 0) * per_trans_order  # Transshipment cost

            hl_cost_total = 0
            order_cost = 0
            # Calculate sum of order, holding, lost sales costs
            for i, retailer in enumerate(self.retailers):  # Iterate through retailers
                retailer.action = action_modified[i + 1]  # Qty ordered by retailer
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
            reward = - trans_cost - hl_cost_total - order_cost

            self.current_period += 1
            if self.current_period >= self.n_period:
                terminate = True
            else:
                terminate = False
            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + L)]  # No +1
            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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
                         retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)
            # the following is the original
            # self.state = np.array(
            #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
            #                                                            retailer.forecast] + \
            #     [x for retailer in self.retailers for x in retailer.pipeline])
        elif len(self.retailers) == 3:
            trans01 = action_modified[0]  # Transshipment quantity, possibly infeasible
            trans02 = action_modified[1]  # Transshipment quantity, possibly infeasible
            trans12 = action_modified[2]  # Transshipment quantity, possibly infeasible
            # Make transshipment quantity feasible
            if trans01 > 0 and self.retailers[0].inv_level < trans01:
                trans01 = 0
            elif trans01 < 0 and self.retailers[1].inv_level < -trans01:
                trans01 = 0
            trans_cost_01 = np.abs(trans01) * per_trans_item + (np.abs(trans01) != 0) * per_trans_order  # Transshipment cost
            if trans02 > 0 and self.retailers[0].inv_level - trans01 < trans02:
                trans02 = 0
            elif trans02 < 0 and self.retailers[2].inv_level < -trans02:
                trans02 = 0
            trans_cost_02 = np.abs(trans02) * per_trans_item + (np.abs(trans02) != 0) * per_trans_order  # Transshipment cost
            if trans12 > 0 and self.retailers[1].inv_level + trans01 < trans12:
                trans12 = 0
            elif trans12 < 0 and self.retailers[2].inv_level + trans02 < -trans12:
                trans12 = 0
            trans_cost_12 = np.abs(trans12) * per_trans_item + (np.abs(trans12) != 0) * per_trans_order  # Transshipment cost
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
            reward = - trans_cost - hl_cost_total - order_cost

            self.current_period += 1
            if self.current_period >= self.n_period:
                terminate = True
            else:
                terminate = False
            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + L)]  # No +1
            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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
                         retailer.transshipment_cost, retailer.fixed_order_transshipment_cost])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)

        return self.state, reward, terminate

    def run_to_get_decision(self, individual): # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0

        decisions = []
        replenishment_decisions = []
        transshipment_decisions = []

        max_ep_len = epi_len  # max timesteps in one episode
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            if len(individual) == 1:
                replenishment_policy = individual[0]
                # ------ get replenishment decision ---------
                decision_replenishment = [state[0]]
                replenishment_decision = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(
                    decision_replenishment)
                replenishment_decisions.append(replenishment_decision)
                # ------ get replenishment decision ---------
            elif len(individual) == 2:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]

                # ------ get replenishment decision ---------
                decision_replenishment = [state[0]]
                replenishment_decision = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(
                    decision_replenishment)
                replenishment_decisions.append(replenishment_decision)
                # ------ get replenishment decision ---------

                # ------ get transshipment decision ---------
                decision_replenishment = [state[1]]
                transshipment_decision = TransshipmentDecisionSituation.TransshipmentDecisionSituation(
                    decision_replenishment)
                transshipment_decisions.append(transshipment_decision)
                # ------ get replenishment decision ---------
            else:
                print("Error in Inventory_simulator_rental_niching.py!")



            # ------- strategy 2 ---------------------
            # quantity_site1 = round(GP_evolve_S(state, replenishment_site1))
            # quantity_site2 = round(GP_evolve_R(state, replenishment_site2))
            #
            # if quantity_site1 < site1_candidate[0]:
            #     quantity_site1 = site1_candidate[0]
            # if quantity_site1 > site1_candidate[len(site1_candidate)-1]:
            #     quantity_site1 = site1_candidate[len(site1_candidate)-1]
            #
            # if quantity_site2 < site2_candidate[0]:
            #     quantity_site2 = site2_candidate[0]
            # if quantity_site2 > site2_candidate[len(site2_candidate)-1]:
            #     quantity_site2 = site2_candidate[len(site2_candidate)-1]
            #
            # action_modified = [0, quantity_site1, quantity_site2]
            # ------- strategy 2 ---------------------

            # # ------- strategy 1 ---------------------
            # # the action space of this one is the same as jinsheng
            # quantity_site1 = GP_evolve_S(state, replenishment_site1)
            # quantity_site2 = GP_evolve_R(state, replenishment_site2)
            #
            # index_site1 = 0
            # min_dis = np.Infinity
            # for i in range(1,len(site1_candidate)):
            #     dis = np.abs(quantity_site1-site1_candidate[i])
            #     if dis < min_dis:
            #         index_site1 = i
            #         min_dis = dis
            #
            # index_site2 = 0
            # min_dis = np.Infinity
            # for i in range(1,len(site2_candidate)):
            #     dis = np.abs(quantity_site2-site2_candidate[i])
            #     if dis < min_dis:
            #         index_site2 = i
            #         min_dis = dis
            #
            # action_modified = [0, site1_candidate[index_site1], site2_candidate[index_site2]]
            # # ------- strategy 1 ---------------------

            # ------- strategy 3 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            action_modified = []
            # get transshipment state for all pairs of sites/retailers
            transshipment_state = state[1]
            replenishment_state = state[0]
            for each_transshipment_state in transshipment_state:
                transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy),2)
                action_modified.append(transshipment_quantity)
            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_evolve_S(each_replenishment_state, replenishment_policy),2)
                action_modified.append(replenishment_quantity)
            # ------- strategy 3 ---------------------

            state, reward, done = self.step_value(action_modified)

            # print("\nsolution, state, reward: " + str(site1_candidate[index_site1]) + ", " + str(state) + ", " + str(reward))

            time_step += 1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        if len(individual) == 1:
            decisions.append(replenishment_decisions)
        elif len(individual) == 2:
            decisions.append(replenishment_decisions)
            decisions.append(transshipment_decisions)
        else:
            print("Error in Inventory_simulator_rental_niching.py!")

        fitness = -current_ep_reward/max_ep_len
        return decisions

