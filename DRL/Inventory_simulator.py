import numpy as np
import time


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
        # todo: modified by mengxu only for the Teckwah that without the second retailer
        # for i in range(len(demand_hist_list[1])):
        #     demand_hist_list[1][i] = 0
        return demand_hist_list


class Retailer:
    def __init__(self, demand_records, number, f,
                 ini_inv, holding, lost_sales, L, LT, capacity, fixed_order,
                 per_trans_item, per_trans_order):  # Each retailer has its own number 0,1,2,... f is its forecast
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
        self.demand_list = demand_records  # Historical demand records
        self.fixed_order_cost = fixed_order[number]  # Fixed cost for placing an order
        self.transshipment_cost = per_trans_item
        self.fixed_order_transshipment_cost = per_trans_order
        self.action = 0  # Order qty

    def reset(self, f):
        self.inv_level = self.ini_inv[self.number]
        self.pipeline = [0] * (self.LT - 1)
        self.forecast = [f(self.number, t) for t in range(1, self.L + 1)]  # Forecast for time t+1

    def order_arrival(self, demand):  # Get next state after pipeline inv arrives and demand is realized
        self.inv_level = min(self.capacity,
                             self.inv_level + self.pipeline[0])  # Pipeline arrives, cannot exceed storage capacity
        self.inv_level -= demand
        # Update pipeline
        self.pipeline = np.concatenate((self.pipeline[1:], [self.action]))

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
        self.lost_sales = 2 * self.holding
        self.capacity = [5 * self.demand_level] * self.num_retailer
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
                retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1

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
            #     retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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
                retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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
            #     retailer.order_arrival(demand[self.current_period - 2])  # -2 not -1
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


