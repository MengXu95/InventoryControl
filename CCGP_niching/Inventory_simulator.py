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

from CCGP_niching.replenishment import *
from CCGP_niching.transshipment import *
from Utils.inventory_core import Retailer, build_demand_model
import threading

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
        self.fixed_order = parameters['fixed_order']
        self.per_trans_item = parameters['per_trans_item']
        self.per_trans_order = parameters['per_trans_order']

        self.rd, self.demand_records = build_demand_model(seed, parameters)
        self.n_retailers = self.num_retailer
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(self.demand_records[i], i, self.rd.f,
                                           self.ini_inv, self.holding, self.lost_sales,
                                           self.L, self.LT, self.capacity, self.fixed_order,
                                           self.per_trans_item, self.per_trans_order))

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
        #the following is the original
        # self.state = np.array(
        #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
        #                                                            retailer.forecast] + \
        #     [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state

    def step_value(self, action_modified):  # modified by mengxu to make it not only suitable for 2 sites
        if len(self.retailers) == 2:
            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 1])  # use zero-based period index

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
                                     range(self.current_period, self.current_period + self.L)]  # No +1
            # # Update inv levels and pipelines
            # for retailer, demand in zip(self.retailers, self.demand_records):
            #     retailer.order_arrival(demand[self.current_period - 1])  # use zero-based period index
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
            return self.state, reward, terminate
            # the following is the original
            # self.state = np.array(
            #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
            #                                                            retailer.forecast] + \
            #     [x for retailer in self.retailers for x in retailer.pipeline])
        elif len(self.retailers) == 3:
            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 1])  # use zero-based period index
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
            reward = - trans_cost - hl_cost_total - order_cost

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
            #     retailer.order_arrival(demand[self.current_period - 1])  # use zero-based period index
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
            return self.state, reward, terminate



    def run(self, individual): # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0

        max_ep_len = self.epi_len  # max timesteps in one episode
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            if len(individual) == 1:
                replenishment_policy = individual[0]
            else:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]

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
            # for i in range(len(site1_candidate)):
            #     dis = np.abs(quantity_site1 - site1_candidate[i])
            #     if dis < min_dis:
            #         index_site1 = i
            #         min_dis = dis
            #
            # action_modified = [0, site1_candidate[index_site1], 0]
            # ------- strategy 1 ---------------------

            # ------- strategy 3 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            action_modified = []
            # get transshipment state for all pairs of sites/retailers
            transshipment_state = state[1]
            replenishment_state = state[0]
            for each_transshipment_state in transshipment_state:
                transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy), 2)
                action_modified.append(transshipment_quantity)
            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_evolve_S(each_replenishment_state, replenishment_policy), 2)
                if replenishment_quantity < 0:
                    replenishment_quantity = 0
                action_modified.append(replenishment_quantity)
            # ------- strategy 3 ---------------------

            # original
            state, reward, done = self.step_value(action_modified)

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

            # break; if the episode is over
            if done:
                break

        fitness = -current_ep_reward/max_ep_len
        return fitness

    def run_test(self, individual, states=None, actions=None, rewards=None): # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0

        max_ep_len = self.epi_len  # max timesteps in one episode
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            if len(individual) == 1:
                replenishment_policy = individual[0]
            else:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]


            # ------- strategy 3 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            action_modified = []
            # get transshipment state for all pairs of sites/retailers
            transshipment_state = state[1]
            replenishment_state = state[0]
            for each_transshipment_state in transshipment_state:
                transshipment_quantity = round(GP_pair_R_test(each_transshipment_state, transshipment_policy),2)
                action_modified.append(transshipment_quantity)
            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_pair_S_test(each_replenishment_state, replenishment_policy),2)
                if replenishment_quantity<0:
                    replenishment_quantity=0
                action_modified.append(replenishment_quantity)
            # ------- strategy 3 ---------------------
            if states is not None:
                states.append(state)

            # the original
            state, reward, done = self.step_value(action_modified)
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

            # break; if the episode is over
            if done:
                break

        fitness = -current_ep_reward/max_ep_len
        return fitness

