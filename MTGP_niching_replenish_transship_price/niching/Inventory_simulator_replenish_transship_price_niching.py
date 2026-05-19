import threading

from MTGP_niching_replenish_transship_price import logistic_util
from MTGP_niching_replenish_transship_price.replenishment import *
from MTGP_niching_replenish_transship_price.transshipment import *
from MTGP_niching_replenish_transship_price.RFQ_price_predict import *
import MTGP_niching_replenish_transship_price.niching.ReplenishmentDecisionSituation as ReplenishmentDecisionSituation
import MTGP_niching_replenish_transship_price.niching.TransshipmentDecisionSituation as TransshipmentDecisionSituation
import MTGP_niching_replenish_transship_price.niching.RFQPredictDecisionSituation as RFQPredictDecisionSituation


# Supplier Support for RFQ pricing
class SupplierSupport:
    def __init__(self, seed, RFQ_demand_max):
        self.seed = seed
        np.random.seed(self.seed)
        self.base_price = 100
        self.alpha = 0.3
        self.beta = 0.5
        self.fluctuation_pct = 0.05
        self.RFQ_demand_max = RFQ_demand_max

    def compute_true_price(self, RFQ_demand, RFQ_time_until_deadline):
        """
        Compute the true price for an RFQ based on demand, deadline, and market fluctuation.
        """
        # Demand effect (normalized by D_max)
        demand_factor = self.alpha * (RFQ_demand / self.RFQ_demand_max)

        # Urgency effect (inverse of deadline + 1 to prevent division by zero)
        urgency_factor = self.beta * (1 / (RFQ_time_until_deadline + 1))

        # Random fluctuation (Gaussian noise)
        fluctuation = np.random.normal(0, self.fluctuation_pct * self.base_price)

        # Compute final price
        P_true = self.base_price * (1 + demand_factor + urgency_factor) + fluctuation

        return P_true


# Demand forecast function with RFQ support
class RandomDemand:
    def __init__(self, seed, demand_level, RFQ_deadline_level, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.demand_level = demand_level
        self.RFQ_deadline_level = RFQ_deadline_level
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.list = np.random.uniform(0, self.demand_level, size=(self.num_retailer, self.epi_len + 3))
        self.deadline_list = np.random.uniform(0, self.RFQ_deadline_level, size=(self.num_retailer, self.epi_len + 3))

    def seedRotation(self):
        self.seed = self.seed + 1000
        np.random.seed(self.seed)

    def reset(self):
        self.seedRotation()
        self.list = np.random.uniform(0, self.demand_level, size=(self.num_retailer, self.epi_len + 3))
        self.deadline_list = np.random.uniform(0, self.RFQ_deadline_level, size=(self.num_retailer, self.epi_len + 3))

    def f(self, n, t):
        if n >= self.num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    def gen_demand(self):
        demand_hist_list = []
        for k in range(self.num_retailer):
            demand_hist = []
            for i in range(1, self.epi_len + 2):
                random_demand = np.random.poisson(self.list[k, i])
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        return demand_hist_list

    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []
        DUT_demand_hist_list = []
        for k in range(self.num_retailer):
            demand_hist = []
            DUT_demand_hist = []
            for i in range(1, self.epi_len + 2):
                random_demand = np.random.uniform(0, self.list[k, i])
                DUT_random_demand = np.random.uniform(0, self.deadline_list[k, i])
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0
                    DUT_random_demand = 0
                demand_hist.append(random_demand)
                DUT_demand_hist.append(DUT_random_demand)
            demand_hist_list.append(demand_hist)
            DUT_demand_hist_list.append(DUT_demand_hist)
        return demand_hist_list, DUT_demand_hist_list


class TeckwahDemand:
    def __init__(self, seed, demand_hist_list, forcast, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.demand_hist_list = demand_hist_list
        self.list = forcast

    def seedRotation(self):
        self.seed = self.seed + 1000
        np.random.seed(self.seed)

    def reset(self):
        self.seedRotation()

    def f(self, n, t):
        if n >= self.num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    def gen_demand(self):
        return self.demand_hist_list

    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []
        DUT_demand_hist_list = []
        for k in range(self.num_retailer):
            demand_hist = []
            DUT_demand_hist = []
            for i in range(1, self.epi_len + 2):
                random_demand = np.random.uniform(0, self.list[k, i])
                DUT_random_demand = 0  # Need to implement deadline for Teckwah
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0
                    DUT_random_demand = 0
                demand_hist.append(random_demand)
                DUT_demand_hist.append(DUT_random_demand)
            demand_hist_list.append(demand_hist)
            DUT_demand_hist_list.append(DUT_demand_hist)
        return demand_hist_list, DUT_demand_hist_list


class Retailer:
    def __init__(self, demand_records, number, f,
                 ini_inv, holding, lost_sales, L, LT, capacity, production_capacity, fixed_order, per_unit_order,
                 per_trans_item, per_trans_order, supplierSupport, RFQ_happen_pro=-1.0):
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
        self.supplierSupport = supplierSupport
        self.RFQ_happen_pro = RFQ_happen_pro

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
        """
        Initialize the inventory optimization environment with replenishment, transshipment, and price prediction.
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
        self.per_unit_order = parameters['per_unit_order']
        self.per_trans_item = parameters['per_trans_item']
        self.per_trans_order = parameters['per_trans_order']
        self.RFQ_deadline_level = parameters['RFQ_deadline_level']
        self.RFQ_happen_pro = parameters['RFQ_happen_pro']
        self.partial_information_visibility = parameters['partial_information_visibility']

        if 'demand_test' in parameters and parameters['demand_test'] is not None:  # use teckwah dataset
            self.demand_records = parameters['demand_test']
            forecast1_all = []
            forecast2_all = []
            for current_period in range(len(self.demand_records[0])):
                forecast1 = [self.demand_records[0, current_period]]
                forecast2 = [self.demand_records[1, current_period]]
                forecast1_all = forecast1_all + forecast1
                forecast2_all = forecast2_all + forecast2
            forecast = np.array([forecast1_all, forecast2_all])
            self.rd = TeckwahDemand(seed, self.demand_records, forecast, self.num_retailer, self.epi_len)
            self.urgent_RFQ_demand_records, self.urgent_RFQ_TUD_records = self.rd.gen_urgent_RFQ_demand(
                self.RFQ_happen_pro)
        else:
            self.rd = RandomDemand(seed, self.demand_level, self.RFQ_deadline_level, self.num_retailer, self.epi_len)
            self.demand_records = self.rd.gen_demand()
            self.urgent_RFQ_demand_records, self.urgent_RFQ_TUD_records = self.rd.gen_urgent_RFQ_demand(
                self.RFQ_happen_pro)

        self.n_retailers = self.num_retailer
        self.retailers = []
        for i in range(self.n_retailers):
            self.supplierSupport = SupplierSupport(seed + i, self.demand_level * 3)
            self.retailers.append(Retailer(self.demand_records[i], i, self.rd.f,
                                           self.ini_inv, self.holding, self.lost_sales,
                                           self.L, self.LT, self.capacity, self.production_capacity, self.fixed_order,
                                           self.per_unit_order,
                                           self.per_trans_item, self.per_trans_order,
                                           self.supplierSupport, self.RFQ_happen_pro))

        self.n_period = len(self.demand_records[0])
        self.current_period = 1

        # Initialize state: [replenishment_state, transshipment_state, RFQ_predict_state]
        self.state = []

        # Replenishment state
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment_retailer = np.array([
                retailer.inv_level, retailer.holding_cost,
                retailer.lost_sales_cost, retailer.capacity,
                retailer.production_capacity,
                retailer.fixed_order_cost, retailer.per_unit_order_cost, retailer.pipeline[0],
                retailer.forecast[0], retailer.forecast[1],
                retailer.transshipment_cost, retailer.fixed_order_transshipment_cost
            ])
            state_replenishment.append(state_replenishment_retailer)
        self.state.append(state_replenishment)

        # Transshipment state
        state_transshipment = []
        for i in range(len(self.retailers)):
            retailer_i = self.retailers[i]
            for j in range(i + 1, len(self.retailers)):
                retailer_j = self.retailers[j]
                state_transshipment_retailer_pair = np.array([
                    i, j,
                    retailer_i.inv_level, retailer_i.holding_cost,
                    retailer_i.lost_sales_cost, retailer_i.capacity,
                    retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                    retailer_i.forecast[0], retailer_i.forecast[1],
                    retailer_j.inv_level, retailer_j.holding_cost,
                    retailer_j.lost_sales_cost, retailer_j.capacity,
                    retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                    retailer_j.forecast[0], retailer_j.forecast[1],
                    retailer_i.transshipment_cost, retailer_i.fixed_order_transshipment_cost
                ])
                state_transshipment.append(state_transshipment_retailer_pair)
        self.state.append(state_transshipment)

        # RFQ prediction state
        state_RFQ_predict = []
        for retailer_index in range(len(self.retailers)):
            state_RFQ_predict_retailer = np.array(
                [self.urgent_RFQ_demand_records[retailer_index][self.current_period - 1],
                 self.urgent_RFQ_TUD_records[retailer_index][self.current_period - 1]])
            state_RFQ_predict.append(state_RFQ_predict_retailer)
        self.state.append(state_RFQ_predict)

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
        self.urgent_RFQ_demand_records, self.urgent_RFQ_TUD_records = self.rd.gen_urgent_RFQ_demand(
            self.RFQ_happen_pro)

        for retailer in self.retailers:
            retailer.reset(self.rd.f)
        self.current_period = 1

        self.state = []

        # Replenishment state
        state_replenishment = []
        for retailer in self.retailers:
            state_replenishment_retailer = np.array([
                retailer.inv_level, retailer.holding_cost,
                retailer.lost_sales_cost, retailer.capacity,
                retailer.production_capacity,
                retailer.fixed_order_cost, retailer.per_unit_order_cost, retailer.pipeline[0],
                retailer.forecast[0], retailer.forecast[1],
                retailer.transshipment_cost, retailer.fixed_order_transshipment_cost
            ])
            state_replenishment.append(state_replenishment_retailer)
        self.state.append(state_replenishment)

        # Transshipment state
        state_transshipment = []
        for i in range(len(self.retailers)):
            retailer_i = self.retailers[i]
            for j in range(i + 1, len(self.retailers)):
                retailer_j = self.retailers[j]
                state_transshipment_retailer_pair = np.array([
                    i, j,
                    retailer_i.inv_level, retailer_i.holding_cost,
                    retailer_i.lost_sales_cost, retailer_i.capacity,
                    retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                    retailer_i.forecast[0], retailer_i.forecast[1],
                    retailer_j.inv_level, retailer_j.holding_cost,
                    retailer_j.lost_sales_cost, retailer_j.capacity,
                    retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                    retailer_j.forecast[0], retailer_j.forecast[1],
                    retailer_j.transshipment_cost, retailer_j.fixed_order_transshipment_cost
                ])
                state_transshipment.append(state_transshipment_retailer_pair)
        self.state.append(state_transshipment)

        # RFQ prediction state
        state_RFQ_predict = []
        for retailer_index in range(len(self.retailers)):
            state_RFQ_predict_retailer = np.array(
                [self.urgent_RFQ_demand_records[retailer_index][self.current_period - 1],
                 self.urgent_RFQ_TUD_records[retailer_index][self.current_period - 1]])
            state_RFQ_predict.append(state_RFQ_predict_retailer)
        self.state.append(state_RFQ_predict)

        return self.state

    def step_value(self, action_modified):
        if len(self.retailers) == 2:
            all_cost = []

            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 1])

            # Handle urgent RFQ demand and price prediction
            total_predict_error = 0
            RFQ_predict_decisions = action_modified[-1]

            if RFQ_predict_decisions != []:
                for RFQ_predict_price, retailer, urgent_RFQ_demand, urgent_RFQ_TUD in zip(
                        RFQ_predict_decisions, self.retailers, self.urgent_RFQ_demand_records,
                        self.urgent_RFQ_TUD_records):
                    if urgent_RFQ_demand[self.current_period - 1] > 0:
                        true_support_price = retailer.supplierSupport.compute_true_price(
                            urgent_RFQ_demand[self.current_period - 1], urgent_RFQ_TUD[self.current_period - 1])
                        predict_support_price = RFQ_predict_price
                        if self.partial_information_visibility:
                            predict_error = np.abs(predict_support_price - true_support_price)
                        else:
                            predict_error = 0
                        total_predict_error += predict_error

            # Handle transshipment
            trans = action_modified[0]
            if trans > 0 and self.retailers[0].inv_level < trans:
                trans = 0
            elif trans < 0 and self.retailers[1].inv_level < -trans:
                trans = 0
            trans_cost = np.abs(trans) * self.per_trans_item + (np.abs(trans) != 0) * self.per_trans_order

            hl_cost_total = 0
            order_cost = 0

            # Calculate order, holding, lost sales costs
            for i, retailer in enumerate(self.retailers):
                retailer.action = action_modified[i + 1]
                if retailer.action > retailer.capacity:
                    retailer.action = retailer.capacity

                order_cost += retailer.action * retailer.per_unit_order_cost + (
                        retailer.action > 0) * retailer.fixed_order_cost

                # Do transshipment
                if retailer.number == 0:
                    retailer.inv_level -= trans
                else:
                    retailer.inv_level += trans

                # Get holding/lost sales cost
                if retailer.inv_level < 0:
                    hl_cost_total += -retailer.inv_level * retailer.lost_sales_cost
                    retailer.inv_level = 0
                else:
                    hl_cost_total += retailer.inv_level * retailer.holding_cost

            reward = -trans_cost - hl_cost_total - order_cost - total_predict_error
            all_cost.append(trans_cost)
            all_cost.append(hl_cost_total)
            all_cost.append(order_cost)
            all_cost.append(total_predict_error)

            self.current_period += 1
            terminate = self.current_period >= self.n_period

            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + self.L)]

            # Update state
            self.state = []

            # Replenishment state
            state_replenishment = []
            for retailer in self.retailers:
                state_replenishment_retailer = np.array([
                    retailer.inv_level, retailer.holding_cost,
                    retailer.lost_sales_cost, retailer.capacity,
                    retailer.production_capacity,
                    retailer.fixed_order_cost, retailer.per_unit_order_cost, retailer.pipeline[0],
                    retailer.forecast[0], retailer.forecast[1],
                    retailer.transshipment_cost, retailer.fixed_order_transshipment_cost
                ])
                state_replenishment.append(state_replenishment_retailer)
            self.state.append(state_replenishment)

            # Transshipment state
            state_transshipment = []
            for i in range(len(self.retailers)):
                retailer_i = self.retailers[i]
                for j in range(i + 1, len(self.retailers)):
                    retailer_j = self.retailers[j]
                    state_transshipment_retailer_pair = np.array([
                        i, j,
                        retailer_i.inv_level, retailer_i.holding_cost,
                        retailer_i.lost_sales_cost, retailer_i.capacity,
                        retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                        retailer_i.forecast[0], retailer_i.forecast[1],
                        retailer_j.inv_level, retailer_j.holding_cost,
                        retailer_j.lost_sales_cost, retailer_j.capacity,
                        retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                        retailer_j.forecast[0], retailer_j.forecast[1],
                        retailer_j.transshipment_cost, retailer_j.fixed_order_transshipment_cost
                    ])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)

            # RFQ prediction state
            state_RFQ_predict = []
            for retailer_index in range(len(self.retailers)):
                state_RFQ_predict_retailer = np.array(
                    [self.urgent_RFQ_demand_records[retailer_index][self.current_period - 1],
                     self.urgent_RFQ_TUD_records[retailer_index][self.current_period - 1]])
                state_RFQ_predict.append(state_RFQ_predict_retailer)
            self.state.append(state_RFQ_predict)

            return self.state, reward, terminate, all_cost

        elif len(self.retailers) == 3:
            all_cost = []

            # Update inv levels and pipelines
            for retailer, demand in zip(self.retailers, self.demand_records):
                retailer.order_arrival(demand[self.current_period - 1])

            # Handle urgent RFQ demand and price prediction
            total_predict_error = 0
            RFQ_predict_decisions = action_modified[-1]

            if RFQ_predict_decisions != []:
                for RFQ_predict_price, retailer, urgent_RFQ_demand, urgent_RFQ_TUD in zip(
                        RFQ_predict_decisions, self.retailers, self.urgent_RFQ_demand_records,
                        self.urgent_RFQ_TUD_records):
                    if urgent_RFQ_demand[self.current_period - 1] > 0:
                        true_support_price = retailer.supplierSupport.compute_true_price(
                            urgent_RFQ_demand[self.current_period - 1], urgent_RFQ_TUD[self.current_period - 1])
                        predict_support_price = RFQ_predict_price
                        if self.partial_information_visibility:
                            predict_error = np.abs(predict_support_price - true_support_price)
                        else:
                            predict_error = 0
                        total_predict_error += predict_error

            # Handle transshipment for 3 retailers
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

            trans_cost = trans_cost_01 + trans_cost_02 + trans_cost_12

            hl_cost_total = 0
            order_cost = 0

            # Calculate order, holding, lost sales costs
            for i, retailer in enumerate(self.retailers):
                retailer.action = action_modified[i + 3]
                if retailer.action > retailer.capacity:
                    retailer.action = retailer.capacity

                order_cost += retailer.action * retailer.per_unit_order_cost + (
                        retailer.action > 0) * retailer.fixed_order_cost

                # Do transshipment
                if retailer.number == 0:
                    retailer.inv_level = retailer.inv_level - trans01 - trans02
                elif retailer.number == 1:
                    retailer.inv_level = retailer.inv_level + trans01 - trans12
                else:
                    retailer.inv_level = retailer.inv_level + trans02 + trans12

                # Get holding/lost sales cost
                if retailer.inv_level < 0:
                    hl_cost_total += -retailer.inv_level * retailer.lost_sales_cost
                    retailer.inv_level = 0
                else:
                    hl_cost_total += retailer.inv_level * retailer.holding_cost

            reward = -trans_cost - hl_cost_total - order_cost - total_predict_error
            all_cost.append(trans_cost)
            all_cost.append(hl_cost_total)
            all_cost.append(order_cost)
            all_cost.append(total_predict_error)

            self.current_period += 1
            terminate = self.current_period >= self.n_period

            # Update forecasts
            for i, retailer in enumerate(self.retailers):
                retailer.forecast = [self.rd.f(i, k) for k in
                                     range(self.current_period, self.current_period + self.L)]

            # Update state
            self.state = []

            # Replenishment state
            state_replenishment = []
            for retailer in self.retailers:
                state_replenishment_retailer = np.array([
                    retailer.inv_level, retailer.holding_cost,
                    retailer.lost_sales_cost, retailer.capacity,
                    retailer.production_capacity,
                    retailer.fixed_order_cost, retailer.per_unit_order_cost, retailer.pipeline[0],
                    retailer.forecast[0], retailer.forecast[1],
                    retailer.transshipment_cost, retailer.fixed_order_transshipment_cost
                ])
                state_replenishment.append(state_replenishment_retailer)
            self.state.append(state_replenishment)

            # Transshipment state
            state_transshipment = []
            for i in range(len(self.retailers)):
                retailer_i = self.retailers[i]
                for j in range(i + 1, len(self.retailers)):
                    retailer_j = self.retailers[j]
                    state_transshipment_retailer_pair = np.array([
                        i, j,
                        retailer_i.inv_level, retailer_i.holding_cost,
                        retailer_i.lost_sales_cost, retailer_i.capacity,
                        retailer_i.fixed_order_cost, retailer_i.pipeline[0],
                        retailer_i.forecast[0], retailer_i.forecast[1],
                        retailer_j.inv_level, retailer_j.holding_cost,
                        retailer_j.lost_sales_cost, retailer_j.capacity,
                        retailer_j.fixed_order_cost, retailer_j.pipeline[0],
                        retailer_j.forecast[0], retailer_j.forecast[1],
                        retailer_j.transshipment_cost, retailer_j.fixed_order_transshipment_cost
                    ])
                    state_transshipment.append(state_transshipment_retailer_pair)
            self.state.append(state_transshipment)

            # RFQ prediction state
            state_RFQ_predict = []
            for retailer_index in range(len(self.retailers)):
                state_RFQ_predict_retailer = np.array(
                    [self.urgent_RFQ_demand_records[retailer_index][self.current_period - 1],
                     self.urgent_RFQ_TUD_records[retailer_index][self.current_period - 1]])
                state_RFQ_predict.append(state_RFQ_predict_retailer)
            self.state.append(state_RFQ_predict)

            return self.state, reward, terminate, all_cost

    def run(self, individual):
        state = self.reset()
        current_ep_reward = 0
        current_ep_all_cost = np.array([0., 0., 0., 0.])

        max_ep_len = self.epi_len
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            if len(individual) == 1:
                replenishment_policy = individual[0]
                transshipment_policy = None
                RFQ_predict_policy = None
            elif len(individual) == 2:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]
                RFQ_predict_policy = None
            elif len(individual) == 3:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]
                RFQ_predict_policy = individual[2]

            action_modified = []
            replenishment_state = state[0]
            transshipment_state = state[1]
            RFQ_predict_state = state[2]

            # Transshipment decisions
            for each_transshipment_state in transshipment_state:
                if transshipment_policy is not None:
                    transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy), 2)
                else:
                    transshipment_quantity = 0
                action_modified.append(transshipment_quantity)

            # Replenishment decisions
            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_evolve_S(each_replenishment_state, replenishment_policy), 2)

                capacity = each_replenishment_state[3]
                upbound_replenishment_quantity = capacity * 3
                if replenishment_quantity > upbound_replenishment_quantity or replenishment_quantity < 0:
                    replenishment_quantity = logistic_util.logistic_scale_and_shift(replenishment_quantity, 0,
                                                                                    upbound_replenishment_quantity)
                action_modified.append(replenishment_quantity)

            # RFQ price prediction decisions
            RFQ_predict_decisions = []
            if RFQ_predict_policy is not None:
                upbound_support_price = self.demand_level * 5
                for each_RFQ_predict_state in RFQ_predict_state:
                    RFQ_predict_price = round(GP_evolve_RFQ_predict(each_RFQ_predict_state, RFQ_predict_policy), 2)
                    if RFQ_predict_price <= 0 or RFQ_predict_price > upbound_support_price:
                        RFQ_predict_price = logistic_util.logistic_scale_and_shift(RFQ_predict_price, 0,
                                                                                   upbound_support_price)
                    RFQ_predict_decisions.append(RFQ_predict_price)

            action_modified.append(RFQ_predict_decisions)

            state, reward, done, all_cost = self.step_value(action_modified)

            time_step += 1
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
        current_ep_all_cost = np.array([0., 0., 0., 0.])

        max_ep_len = self.epi_len
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            if len(individual) == 1:
                replenishment_policy = individual[0]
                transshipment_policy = None
                RFQ_predict_policy = None
            elif len(individual) == 2:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]
                RFQ_predict_policy = None
            elif len(individual) == 3:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]
                RFQ_predict_policy = individual[2]

            action_modified = []
            replenishment_state = state[0]
            transshipment_state = state[1]
            RFQ_predict_state = state[2]

            # Transshipment decisions
            for each_transshipment_state in transshipment_state:
                if transshipment_policy is not None:
                    transshipment_quantity = round(GP_pair_R_test(each_transshipment_state, transshipment_policy), 2)
                else:
                    transshipment_quantity = 0
                action_modified.append(transshipment_quantity)

            # Replenishment decisions
            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_pair_S_test(each_replenishment_state, replenishment_policy), 2)

                capacity = each_replenishment_state[3]
                upbound_replenishment_quantity = capacity * 3
                if replenishment_quantity > upbound_replenishment_quantity or replenishment_quantity < 0:
                    replenishment_quantity = logistic_util.logistic_scale_and_shift(replenishment_quantity, 0,
                                                                                    upbound_replenishment_quantity)
                action_modified.append(replenishment_quantity)

            # RFQ price prediction decisions
            RFQ_predict_decisions = []
            if RFQ_predict_policy is not None:
                upbound_support_price = self.demand_level * 5
                for each_RFQ_predict_state in RFQ_predict_state:
                    RFQ_predict_price = round(GP_pair_RFQ_predict_test(each_RFQ_predict_state, RFQ_predict_policy), 2)
                    if RFQ_predict_price <= 0 or RFQ_predict_price > upbound_support_price:
                        RFQ_predict_price = logistic_util.logistic_scale_and_shift(RFQ_predict_price, 0,
                                                                                   upbound_support_price)
                    RFQ_predict_decisions.append(RFQ_predict_price)

            action_modified.append(RFQ_predict_decisions)

            if states is not None:
                states.append(state)

            state, reward, done, all_cost = self.step_value(action_modified)

            if actions is not None:
                actions.append(action_modified)
            if rewards is not None:
                rewards.append(reward)

            time_step += 1
            current_ep_reward += reward
            current_ep_all_cost += np.array(all_cost)

            if done:
                break

        fitness = -current_ep_reward / max_ep_len
        all_cost_fit = current_ep_all_cost / max_ep_len
        return fitness, all_cost_fit

    def run_to_get_decision(self, individual):
        """
        Run simulation and collect decision points for niching.
        Returns decision situations for replenishment, transshipment, and RFQ prediction.
        """
        state = self.reset()
        current_ep_reward = 0
        current_ep_all_cost = np.array([0., 0., 0., 0.])

        decisions = []
        replenishment_decision_points = []
        transshipment_decision_points = []
        RFQ_predict_decision_points = []

        max_ep_len = self.epi_len
        time_step = 0

        for _ in range(1, max_ep_len + 1):
            # Determine policies based on individual structure
            if len(individual) == 1:
                replenishment_policy = individual[0]
                transshipment_policy = None
                RFQ_predict_policy = None

                # Collect replenishment decision points
                decision_replenishment = [state[0]]
                replenishment_decision_point = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(
                    decision_replenishment)
                replenishment_decision_points.append(replenishment_decision_point)

            elif len(individual) == 2:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]
                RFQ_predict_policy = None

                # Collect replenishment decision points
                decision_replenishment = [state[0]]
                replenishment_decision_point = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(
                    decision_replenishment)
                replenishment_decision_points.append(replenishment_decision_point)

                # Collect transshipment decision points
                decision_transshipment = [state[1]]
                transshipment_decision_point = TransshipmentDecisionSituation.TransshipmentDecisionSituation(
                    decision_transshipment)
                transshipment_decision_points.append(transshipment_decision_point)

            elif len(individual) == 3:
                replenishment_policy = individual[0]
                transshipment_policy = individual[1]
                RFQ_predict_policy = individual[2]

                # Collect replenishment decision points
                decision_replenishment = [state[0]]
                replenishment_decision_point = ReplenishmentDecisionSituation.ReplenishmentDecisionSituation(
                    decision_replenishment)
                replenishment_decision_points.append(replenishment_decision_point)

                # Collect transshipment decision points
                decision_transshipment = [state[1]]
                transshipment_decision_point = TransshipmentDecisionSituation.TransshipmentDecisionSituation(
                    decision_transshipment)
                transshipment_decision_points.append(transshipment_decision_point)

                # Collect RFQ prediction decision points
                decision_RFQ_predict = [state[2]]
                RFQ_predict_decision_point = RFQPredictDecisionSituation.RFQPredictDecisionSituation(
                    decision_RFQ_predict)
                RFQ_predict_decision_points.append(RFQ_predict_decision_point)

            # Execute actions
            action_modified = []
            replenishment_state = state[0]
            transshipment_state = state[1]
            RFQ_predict_state = state[2]

            # Transshipment decisions
            for each_transshipment_state in transshipment_state:
                if transshipment_policy is not None:
                    transshipment_quantity = round(GP_evolve_R(each_transshipment_state, transshipment_policy), 2)
                else:
                    transshipment_quantity = 0
                action_modified.append(transshipment_quantity)

            # Replenishment decisions
            for each_replenishment_state in replenishment_state:
                replenishment_quantity = round(GP_evolve_S(each_replenishment_state, replenishment_policy), 2)

                capacity = each_replenishment_state[3]
                upbound_replenishment_quantity = capacity * 3
                if replenishment_quantity > upbound_replenishment_quantity or replenishment_quantity < 0:
                    replenishment_quantity = logistic_util.logistic_scale_and_shift(replenishment_quantity, 0,
                                                                                    upbound_replenishment_quantity)
                action_modified.append(replenishment_quantity)

            # RFQ price prediction decisions
            RFQ_predict_decisions = []
            if RFQ_predict_policy is not None:
                upbound_support_price = self.demand_level * 5
                for each_RFQ_predict_state in RFQ_predict_state:
                    RFQ_predict_price = round(GP_evolve_RFQ_predict(each_RFQ_predict_state, RFQ_predict_policy), 2)
                    if RFQ_predict_price <= 0 or RFQ_predict_price > upbound_support_price:
                        RFQ_predict_price = logistic_util.logistic_scale_and_shift(RFQ_predict_price, 0,
                                                                                   upbound_support_price)
                    RFQ_predict_decisions.append(RFQ_predict_price)

            action_modified.append(RFQ_predict_decisions)

            # Step environment
            state, reward, done, all_cost = self.step_value(action_modified)

            time_step += 1
            current_ep_reward += reward
            current_ep_all_cost += np.array(all_cost)

            if done:
                break

        # Collect all decision points based on individual structure
        if len(individual) == 1:
            decisions.append(replenishment_decision_points)
        elif len(individual) == 2:
            decisions.append(replenishment_decision_points)
            decisions.append(transshipment_decision_points)
        elif len(individual) == 3:
            decisions.append(replenishment_decision_points)
            decisions.append(transshipment_decision_points)
            decisions.append(RFQ_predict_decision_points)
        else:
            print("Error in Inventory_simulator_transship_RFQ_price_niching.py!")

        return decisions