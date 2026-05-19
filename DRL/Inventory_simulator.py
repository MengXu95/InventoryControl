import numpy as np
import time
from Utils.inventory_core import RandomDemand, Retailer

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

        self.rd = RandomDemand(seed, self.demand_level, self.num_retailer, self.epi_len)
        self.demand_records = self.rd.gen_demand()
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
            state_replenishment_retailer = [
                retailer.inv_level, retailer.holding_cost,
                retailer.lost_sales_cost, retailer.capacity,
                retailer.fixed_order_cost, retailer.pipeline[0],  # only suitable for LT = 2
                retailer.forecast[0], retailer.forecast[1]]  # only suitable for LT = 2
            state_replenishment = state_replenishment + state_replenishment_retailer
        self.state = np.array(state_replenishment + [self.retailers[0].transshipment_cost, self.retailers[0].fixed_order_transshipment_cost])


    def reset(self):  # Resets state of all retailers and DCs by calling their respective reset methods
        self.rd.reset()
        self.demand_records = self.rd.gen_demand()  # New instance of demand realizations
        for retailer in self.retailers:
            retailer.reset(self.rd.f)
        self.current_period = 1
        self.state = []  # include replenishment state of each retailer and transshipment state of each pair of sites
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment_retailer = [retailer.inv_level, retailer.holding_cost,
                                                     retailer.lost_sales_cost, retailer.capacity,
                                                     retailer.fixed_order_cost, retailer.pipeline[0],
                                                     # only suitable for LT = 2
                                                     retailer.forecast[0], retailer.forecast[1]]  # only suitable for LT = 2
            state_replenishment = state_replenishment + state_replenishment_retailer
        self.state = np.array(state_replenishment + [self.retailers[0].transshipment_cost, self.retailers[0].fixed_order_transshipment_cost])
        #the following is the original
        # self.state = np.array(
        #     [retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in
        #                                                            retailer.forecast] + \
        #     [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state

    def step(self, action_map, action):  # modified by mengxu to make it not only suitable for 2 sites
        action_modified = action_map[action]
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
                state_replenishment_retailer = [retailer.inv_level, retailer.holding_cost,
                                                         retailer.lost_sales_cost, retailer.capacity,
                                                         retailer.fixed_order_cost, retailer.pipeline[0],
                                                         # only suitable for LT = 2
                                                         retailer.forecast[0], retailer.forecast[1]]  # only suitable for LT = 2
                state_replenishment = state_replenishment + state_replenishment_retailer
            self.state = np.array(state_replenishment + [self.retailers[0].transshipment_cost, self.retailers[0].fixed_order_transshipment_cost])
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
                state_replenishment_retailer = [retailer.inv_level, retailer.holding_cost,
                                                         retailer.lost_sales_cost, retailer.capacity,
                                                         retailer.fixed_order_cost, retailer.pipeline[0],
                                                         # only suitable for LT = 2
                                                         retailer.forecast[0], retailer.forecast[1]]  # only suitable for LT = 2
                state_replenishment = state_replenishment + state_replenishment_retailer
            self.state = np.array(state_replenishment + [self.retailers[0].transshipment_cost,self.retailers[0].fixed_order_transshipment_cost])
            return self.state, reward, terminate

    def run_test(self, ppo_agent, action_map, states=None, actions=None, rewards=None):  # add by xumeng 2024.8.1
        # run simulation
        state = self.reset()
        current_ep_reward = 0

        max_ep_len = self.epi_len  # max timesteps in one episode
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # select action with policy

            # ------- strategy 3 ---------------------
            # the action space of this one is the same as jinsheng
            # for the scenario that only consider one site
            action = ppo_agent.top_action(state)
            action_modified = action_map[action]
            start = time.time()

            end = time.time()
            running_time = end - start
            # print("time:" + str(running_time))
            # ------- strategy 3 ---------------------
            if states is not None:
                states.append(state)
            # the original
            state, reward, done = self.step(action_map, action)

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
        if time_step < max_ep_len:
            return np.inf
        fitness = -current_ep_reward / max_ep_len
        return fitness


