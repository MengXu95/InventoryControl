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

from MTGP_new_terminals.sequencing import *
from MTGP_new_terminals.routing import *

# np.random.seed(0)
###############training for Teckwah##########################################
# Parameters
L = 2 # Length of forecast horizon
LT = 2 # Lead time
epi_len = 64 # Length of one episode
ini_inv = [10000,0] # Initial inventory levels
holding = [2,0] # Holding costs
lost_sales = [50,0] # Per unit lost sales costs
capacity = [200000,0] # Inventory capacities
fixed_order = [10000,0] # Fixed order costs per order
per_trans_item = 0 # Per unit cost for transshipment (either direction)
per_trans_order = 5 # Fixed cost per transshipment (either direction)
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

# # Demand forecast function
# # Note that at decision time t, demand for time t has already been realised
class RandomDemand:
    def __init__(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.list = np.random.uniform(0, 30000, size=(2, epi_len + 3)) # for Teckwah
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        for i in range(len(self.list[1])):
            self.list[1][i] = 0
        # self.list = np.random.uniform(0, 15, size=(2, epi_len + 3))

    def seedRotation(self): # add by xumeng for changing to a new seed
        self.seed = self.seed + 1000
        np.random.seed(self.seed)
    def reset(self):
        self.seedRotation() # add by xumeng for changing to a new seed
        self.list = np.random.uniform(0, 30000, size=(2, epi_len + 3))# for Teckwah
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        for i in range(len(self.list[1])):
            self.list[1][i] = 0
        # self.list = np.random.uniform(0, 15, size=(2, epi_len + 3))

    def f(self, n, t):  # Generate forecasts, f(n,t) corresponds to demand mean for retailer n at time t+1
        if n > 1:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    # Function to generate demand history for the two retailers, of length epi_len+1
    def gen_demand(self):
        demand_hist_list = []  # List to hold demand histories for multiple retailers
        for k in range(2):
            demand_hist = []
            for i in range(1, epi_len + 2):  # 1 extra demand generated so that last state has a next state
                random_demand = np.random.poisson(self.list[k, i])  # Poisson distribution with forecasted mean
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        for i in range(len(demand_hist_list[1])):
            demand_hist_list[1][i] = 0
        return demand_hist_list


# def forec(n, t): # Demand forecast for retailer n=0,1 at time t=1,2,...
#     if n==0:
#       k=4*math.sin(math.pi*t/8)+6 # Not rounded, so non-integer
#     elif n==1:
#       k=4*math.cos(math.pi*t/8)+6
#     else:
#       raise ValueError("Invalid retailer number")
#     return k


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
        self.n_retailers = 2
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(self.demand_records[i], i, self.rd.f))
        self.n_period = len(self.demand_records[0])
        self.current_period = 1
        self.state = np.array(
            [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
                                                                   retailer.forecast] + \
            [x for retailer in self.retailers for x in retailer.pipeline])

    def reset(self):  # Resets state of all retailers and DCs by calling their respective reset methods
        self.rd.reset()
        self.demand_records = self.rd.gen_demand()  # New instance of demand realizations
        for retailer in self.retailers:
            retailer.reset(self.rd.f)
        self.current_period = 1

        self.state = []
        for i in range(len(self.retailers)):
            state_each = np.array([self.retailers[i].inv_level] + [x for x in self.retailers[i].forecast] + \
                                  [x for x in self.retailers[i].pipeline] + [holding[i]] + [lost_sales[i]] + \
                                  [fixed_order[i]])
            self.state.append(state_each)

        # self.state = np.array(
        #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
        #                                                            retailer.forecast] + \
        #     [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state

    def step(self, action):
        action_modified = action_map[action]
        trans = action_modified[0]  # Transshipment quantity, possibly infeasible
        # Make transshipment quantity feasible
        if trans > 0 and self.retailers[0].inv_level < trans:
            trans = 0
        elif trans < 0 and self.retailers[1].inv_level < -trans:
            trans = 0
        trans_cost = trans * per_trans_item + (trans != 0) * per_trans_order  # Transshipment cost

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
            retailer.forecast = [self.rd.f(i, k) for k in range(self.current_period, self.current_period + L)]  # No +1
        # Update inv levels and pipelines
        for retailer, demand in zip(self.retailers, self.demand_records):
            retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
        self.state = []
        for i in range(len(self.retailers)):
            state_each = np.array([self.retailers[i].inv_level] + [x for x in self.retailers[i].forecast] + \
                                  [x for x in self.retailers[i].pipeline] + [holding[i]] + [lost_sales[i]] + \
                                  [fixed_order[i]])
            self.state.append(state_each)
        return self.state, reward, terminate

    def step_value(self, action_modified):
        trans = action_modified[0]  # Transshipment quantity, possibly infeasible
        # Make transshipment quantity feasible
        if trans > 0 and self.retailers[0].inv_level < trans:
            trans = 0
        elif trans < 0 and self.retailers[1].inv_level < -trans:
            trans = 0
        trans_cost = trans * per_trans_item + (trans != 0) * per_trans_order  # Transshipment cost

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
            retailer.forecast = [self.rd.f(i, k) for k in range(self.current_period, self.current_period + L)]  # No +1
        # Update inv levels and pipelines
        for retailer, demand in zip(self.retailers, self.demand_records):
            retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
        self.state = []
        for i in range(len(self.retailers)):
            state_each = np.array([self.retailers[i].inv_level] + [x for x in self.retailers[i].forecast] + \
                                  [x for x in self.retailers[i].pipeline] + [holding[i]] + [lost_sales[i]] + \
                                  [fixed_order[i]])
            self.state.append(state_each)
        return self.state, reward, terminate

    def run(self, individual): # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0

        max_ep_len = epi_len  # max timesteps in one episode
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            # if len(individual) == 1:
            #     site1_candidate = action_lists[1]
            #     replenishment_site1 = individual[0]
            # else:
            #     site1_candidate = action_lists[1]
            #     site2_candidate = action_lists[2]
            #     replenishment_site1 = individual[0]
            #     replenishment_site2 = individual[1]

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

            # ------- strategy 1 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            # quantity_site1 = GP_evolve_S(state, replenishment_site1)
            #
            # index_site1 = 0
            # min_dis = np.Infinity
            # for i in range(1, len(site1_candidate)):
            #     dis = np.abs(quantity_site1 - site1_candidate[i])
            #     if dis < min_dis:
            #         index_site1 = i
            #         min_dis = dis
            #
            # action_modified = [0, site1_candidate[index_site1], 0]
            # ------- strategy 1 ---------------------

            action_modified = [0]
            replenishment_site1 = individual[0]

            for s in range(len(state)):
                state_each = state[s]
                site1_candidate = action_lists[s+1]

                quantity_site1 = GP_evolve_S(state_each, replenishment_site1)

                index_site1 = 0
                min_dis = np.Infinity
                for i in range(len(site1_candidate)):
                    dis = np.abs(quantity_site1 - site1_candidate[i])
                    if dis < min_dis:
                        index_site1 = i
                        min_dis = dis
                action_modified.append(site1_candidate[index_site1])

            action_modified[len(action_modified)-1] = 0

            state, reward, done = self.step_value(action_modified)

            # print("\nsolution, state, reward: " + str(site1_candidate[index_site1]) + ", " + str(state) + ", " + str(reward))

            time_step += 1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        fitness = -current_ep_reward/max_ep_len
        return fitness

