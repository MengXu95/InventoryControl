import threading

import numpy as np

from MTGP_niching_replenish_transship.replenishment import *
from MTGP_niching_replenish_transship.transshipment import *
from Utils.inventory_core import build_demand_model
import MTGP_niching_replenish_transship.logistic_util as logistic_util
import MTGP_niching_replenish_transship.niching.ReplenishmentDecisionSituation as ReplenishmentDecisionSituation
import MTGP_niching_replenish_transship.niching.TransshipmentDecisionSituation as TransshipmentDecisionSituation


class Retailer:
    def __init__(self, demand_records, number, f,
                 ini_inv, holding, lost_sales, L, LT, capacity, production_capacity, fixed_order, per_unit_order,
                 per_trans_item, per_trans_order):
        self.ini_inv = ini_inv
        self.L = L
        self.LT = LT
        self.number = number
        self.inv_level = ini_inv[number]
        self.holding_cost = holding[number]
        self.lost_sales_cost = lost_sales[number]
        self.pipeline = [0] * (LT - 1)
        self.forecast = [f(number, t) for t in range(1, L + 1)]
        self.capacity = capacity[number]
        self.production_capacity = production_capacity[number]
        self.demand_list = demand_records
        self.fixed_order_cost = fixed_order[number]
        self.per_unit_order_cost = per_unit_order[number]
        self.transshipment_cost = per_trans_item
        self.fixed_order_transshipment_cost = per_trans_order
        self.action = 0

    def reset(self, f):
        self.inv_level = self.ini_inv[self.number]
        self.pipeline = [0] * (self.LT - 1)
        self.forecast = [f(self.number, t) for t in range(1, self.L + 1)]

    def order_arrival(self, demand):
        self.inv_level = min(self.capacity, self.inv_level + self.pipeline[0])
        self.inv_level -= demand
        self.pipeline = np.concatenate((self.pipeline[1:], [self.action]))


class TimeoutException(Exception):
    pass


class InvOptEnv:
    def __init__(self, seed, parameters):
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
        self.per_unit_order = parameters['per_unit_order']
        self.per_trans_item = parameters['per_trans_item']
        self.per_trans_order = parameters['per_trans_order']

        self.rd, self.demand_records = build_demand_model(seed, parameters)
        self.n_retailers = self.num_retailer
        self.retailers = []
        for retailer_index in range(self.n_retailers):
            self.retailers.append(Retailer(
                self.demand_records[retailer_index], retailer_index, self.rd.f,
                self.ini_inv, self.holding, self.lost_sales,
                self.L, self.LT, self.capacity, self.production_capacity, self.fixed_order,
                self.per_unit_order, self.per_trans_item, self.per_trans_order,
            ))

        self.n_period = len(self.demand_records[0])
        self.current_period = 1
        self.state = self._build_state()

    def _build_replenishment_state(self):
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment.append(np.array([
                retailer.inv_level, retailer.holding_cost,
                retailer.lost_sales_cost, retailer.capacity,
                retailer.production_capacity,
                retailer.fixed_order_cost, retailer.per_unit_order_cost, retailer.pipeline[0],
                retailer.forecast[0], retailer.forecast[1],
                retailer.transshipment_cost, retailer.fixed_order_transshipment_cost,
            ]))
        return state_replenishment

    def _build_transshipment_state(self):
        state_transshipment = []
        for i in range(len(self.retailers)):
            retailer_i = self.retailers[i]
            for j in range(i + 1, len(self.retailers)):
                retailer_j = self.retailers[j]
                state_transshipment.append(np.array([
                    i, j,
                    retailer_i.inv_level, retailer_i.holding_cost,
                    retailer_i.lost_sales_cost, retailer_i.capacity,
                    retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                    retailer_i.forecast[0], retailer_i.forecast[1],
                    retailer_j.inv_level, retailer_j.holding_cost,
                    retailer_j.lost_sales_cost, retailer_j.capacity,
                    retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                    retailer_j.forecast[0], retailer_j.forecast[1],
                    retailer_i.transshipment_cost, retailer_i.fixed_order_transshipment_cost,
                ]))
        return state_transshipment

    def _build_state(self):
        return [self._build_replenishment_state(), self._build_transshipment_state()]

    def timeout_handler(self):
        raise TimeoutException("Operation timed out!")

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        timer = threading.Timer(timeout, self.timeout_handler)
        timer.start()
        try:
            result = func(*args, **kwargs)
        except TimeoutException:
            print("Function took too long to run!")
            result = np.nan
        finally:
            timer.cancel()
        return result

    def reset(self):
        self.rd.reset()
        self.demand_records = self.rd.gen_demand()
        for retailer in self.retailers:
            retailer.reset(self.rd.f)
        self.current_period = 1
        self.state = self._build_state()
        return self.state

    def _apply_two_site_transshipments(self, action_modified):
        trans = action_modified[0]
        if trans > 0 and self.retailers[0].inv_level < trans:
            trans = 0
        elif trans < 0 and self.retailers[1].inv_level < -trans:
            trans = 0

        trans_cost = np.abs(trans) * self.per_trans_item + (np.abs(trans) != 0) * self.per_trans_order
        self.retailers[0].inv_level -= trans
        self.retailers[1].inv_level += trans
        return trans_cost

    def _apply_three_site_transshipments(self, action_modified):
        trans01 = action_modified[0]
        trans02 = action_modified[1]
        trans12 = action_modified[2]

        if trans01 > 0 and self.retailers[0].inv_level < trans01:
            trans01 = 0
        elif trans01 < 0 and self.retailers[1].inv_level < -trans01:
            trans01 = 0
        trans_cost_01 = np.abs(trans01) * self.per_trans_item + (np.abs(trans01) != 0) * self.per_trans_order

        if trans02 > 0 and self.retailers[0].inv_level - trans01 < trans02:
            trans02 = 0
        elif trans02 < 0 and self.retailers[2].inv_level < -trans02:
            trans02 = 0
        trans_cost_02 = np.abs(trans02) * self.per_trans_item + (np.abs(trans02) != 0) * self.per_trans_order

        if trans12 > 0 and self.retailers[1].inv_level + trans01 < trans12:
            trans12 = 0
        elif trans12 < 0 and self.retailers[2].inv_level + trans02 < -trans12:
            trans12 = 0
        trans_cost_12 = np.abs(trans12) * self.per_trans_item + (np.abs(trans12) != 0) * self.per_trans_order

        self.retailers[0].inv_level = self.retailers[0].inv_level - trans01 - trans02
        self.retailers[1].inv_level = self.retailers[1].inv_level + trans01 - trans12
        self.retailers[2].inv_level = self.retailers[2].inv_level + trans02 + trans12
        return trans_cost_01 + trans_cost_02 + trans_cost_12

    def step_value(self, action_modified):
        all_cost = []
        for retailer, demand in zip(self.retailers, self.demand_records):
            retailer.order_arrival(demand[self.current_period - 1])

        if len(self.retailers) == 2:
            trans_cost = self._apply_two_site_transshipments(action_modified)
            order_offset = 1
        elif len(self.retailers) == 3:
            trans_cost = self._apply_three_site_transshipments(action_modified)
            order_offset = 3
        else:
            raise ValueError("Only 2-site and 3-site replenish/transship scenarios are currently supported.")

        hl_cost_total = 0
        order_cost = 0
        for retailer_index, retailer in enumerate(self.retailers):
            retailer.action = action_modified[retailer_index + order_offset]
            if retailer.action > retailer.capacity:
                retailer.action = retailer.capacity

            order_cost += retailer.action * retailer.per_unit_order_cost + (retailer.action > 0) * retailer.fixed_order_cost

            if retailer.inv_level < 0:
                hl_cost_total += -retailer.inv_level * retailer.lost_sales_cost
                retailer.inv_level = 0
            else:
                hl_cost_total += retailer.inv_level * retailer.holding_cost

        reward = -trans_cost - hl_cost_total - order_cost
        all_cost.append(trans_cost)
        all_cost.append(hl_cost_total)
        all_cost.append(order_cost)

        self.current_period += 1
        terminate = self.current_period >= self.n_period

        for retailer_index, retailer in enumerate(self.retailers):
            retailer.forecast = [self.rd.f(retailer_index, period_index)
                                 for period_index in range(self.current_period, self.current_period + self.L)]

        self.state = self._build_state()
        return self.state, reward, terminate, all_cost

    def _build_action(self, state, replenishment_policy, transshipment_policy, use_test=False):
        action_modified = []
        replenishment_state = state[0]
        transshipment_state = state[1]

        for each_transshipment_state in transshipment_state:
            if transshipment_policy is not None:
                if use_test:
                    transshipment_quantity = round(GP_pair_R_test(each_transshipment_state, transshipment_policy), 2)
                else:
                    transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy), 2)
            else:
                transshipment_quantity = 0
            action_modified.append(transshipment_quantity)

        for each_replenishment_state in replenishment_state:
            if use_test:
                replenishment_quantity = round(GP_pair_S_test(each_replenishment_state, replenishment_policy), 2)
            else:
                replenishment_quantity = round(GP_evolve_S(each_replenishment_state, replenishment_policy), 2)

            capacity = each_replenishment_state[3]
            upbound_replenishment_quantity = capacity * 3
            if replenishment_quantity > upbound_replenishment_quantity or replenishment_quantity < 0:
                replenishment_quantity = logistic_util.logistic_scale_and_shift(
                    replenishment_quantity, 0, upbound_replenishment_quantity)
            action_modified.append(replenishment_quantity)

        return action_modified

    def run(self, individual):
        state = self.reset()
        current_ep_reward = 0
        current_ep_all_cost = np.array([0., 0., 0.])
        max_ep_len = self.epi_len

        for _ in range(1, max_ep_len + 1):
            replenishment_policy = individual[0]
            transshipment_policy = individual[1] if len(individual) > 1 else None
            action_modified = self._build_action(state, replenishment_policy, transshipment_policy)
            state, reward, done, all_cost = self.step_value(action_modified)
            current_ep_reward += reward
            current_ep_all_cost += np.array(all_cost)
            if done:
                break

        fitness = -current_ep_reward / max_ep_len
        all_cost_fit = current_ep_all_cost / max_ep_len
        return fitness, all_cost_fit

    def run_test(self, individual, states=None, actions=None, rewards=None):
        state = self.reset()
        current_ep_reward = 0
        current_ep_all_cost = np.array([0., 0., 0.])
        max_ep_len = self.epi_len

        for _ in range(1, max_ep_len + 1):
            replenishment_policy = individual[0]
            transshipment_policy = individual[1] if len(individual) > 1 else None
            action_modified = self._build_action(state, replenishment_policy, transshipment_policy, use_test=True)

            if states is not None:
                states.append(state)

            state, reward, done, all_cost = self.step_value(action_modified)

            if actions is not None:
                actions.append(action_modified)
            if rewards is not None:
                rewards.append(reward)

            current_ep_reward += reward
            current_ep_all_cost += np.array(all_cost)
            if done:
                break

        fitness = -current_ep_reward / max_ep_len
        all_cost_fit = current_ep_all_cost / max_ep_len
        return fitness, all_cost_fit

    def run_to_get_decision(self, individual):
        state = self.reset()
        replenishment_decision_points = []
        transshipment_decision_points = []
        max_ep_len = self.epi_len

        for _ in range(1, max_ep_len + 1):
            replenishment_policy = individual[0]
            transshipment_policy = individual[1] if len(individual) > 1 else None

            replenishment_decision_points.append(
                ReplenishmentDecisionSituation.ReplenishmentDecisionSituation([state[0]]))
            if transshipment_policy is not None:
                transshipment_decision_points.append(
                    TransshipmentDecisionSituation.TransshipmentDecisionSituation([state[1]]))

            action_modified = self._build_action(state, replenishment_policy, transshipment_policy)
            state, reward, done, all_cost = self.step_value(action_modified)
            if done:
                break

        decisions = [replenishment_decision_points]
        if len(individual) > 1:
            decisions.append(transshipment_decision_points)
        return decisions
