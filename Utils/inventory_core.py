import numpy as np


class RandomDemand:
    def __init__(self, seed, demand_level, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.demand_level = demand_level
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.list = np.random.uniform(0, self.demand_level, size=(self.num_retailer, self.epi_len + 3))

    def seedRotation(self):
        self.seed = self.seed + 1000
        np.random.seed(self.seed)

    def reset(self):
        self.seedRotation()
        self.list = np.random.uniform(0, self.demand_level, size=(self.num_retailer, self.epi_len + 3))

    def f(self, n, t):
        if n >= self.num_retailer:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    def gen_demand(self):
        demand_hist_list = []
        for k in range(self.num_retailer):
            demand_hist = []
            for i in range(1, self.epi_len + 2):
                demand_hist.append(np.random.poisson(self.list[k, i]))
            demand_hist_list.append(demand_hist)
        return demand_hist_list


class TeckwahDemand:
    def __init__(self, seed, demand_hist_list, forecast, num_retailer, epi_len):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_retailer = num_retailer
        self.epi_len = epi_len
        self.demand_hist_list = demand_hist_list
        self.list = forecast

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


def build_teckwah_forecast(demand_records):
    return np.array([list(demand_records[retailer_index]) for retailer_index in range(len(demand_records))])


def build_demand_model(seed, parameters):
    if 'demand_test' in parameters and parameters['demand_test'] is not None:
        demand_records = parameters['demand_test']
        forecast = build_teckwah_forecast(demand_records)
        demand_model = TeckwahDemand(
            seed,
            demand_records,
            forecast,
            parameters['num_retailer'],
            parameters['epi_len'],
        )
    else:
        demand_model = RandomDemand(
            seed,
            parameters['demand_level'],
            parameters['num_retailer'],
            parameters['epi_len'],
        )
        demand_records = demand_model.gen_demand()
    return demand_model, demand_records


class UrgentRFQRandomDemand(RandomDemand):
    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []
        for retailer_index in range(self.num_retailer):
            demand_hist = []
            for period_index in range(1, self.epi_len + 2):
                random_demand = np.random.uniform(0, self.list[retailer_index, period_index])
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        return demand_hist_list


class UrgentRFQTeckwahDemand(TeckwahDemand):
    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []
        for retailer_index in range(self.num_retailer):
            demand_hist = []
            for period_index in range(1, self.epi_len + 2):
                random_demand = np.random.uniform(0, self.list[retailer_index, period_index])
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        return demand_hist_list


class DeadlineRFQRandomDemand(RandomDemand):
    def __init__(self, seed, demand_level, RFQ_deadline_level, num_retailer, epi_len):
        super().__init__(seed, demand_level, num_retailer, epi_len)
        self.RFQ_deadline_level = RFQ_deadline_level
        self.deadline_list = np.random.uniform(0, self.RFQ_deadline_level, size=(self.num_retailer, self.epi_len + 3))

    def reset(self):
        super().reset()
        self.deadline_list = np.random.uniform(0, self.RFQ_deadline_level, size=(self.num_retailer, self.epi_len + 3))

    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []
        deadline_hist_list = []
        for retailer_index in range(self.num_retailer):
            demand_hist = []
            deadline_hist = []
            for period_index in range(1, self.epi_len + 2):
                random_demand = np.random.uniform(0, self.list[retailer_index, period_index])
                random_deadline = np.random.uniform(0, self.deadline_list[retailer_index, period_index])
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0
                    random_deadline = 0
                demand_hist.append(random_demand)
                deadline_hist.append(random_deadline)
            demand_hist_list.append(demand_hist)
            deadline_hist_list.append(deadline_hist)
        return demand_hist_list, deadline_hist_list


class DeadlineRFQTeckwahDemand(TeckwahDemand):
    def gen_urgent_RFQ_demand(self, RFQ_happen_pro):
        demand_hist_list = []
        deadline_hist_list = []
        for retailer_index in range(self.num_retailer):
            demand_hist = []
            deadline_hist = []
            for period_index in range(1, self.epi_len + 2):
                random_demand = np.random.uniform(0, self.list[retailer_index, period_index])
                random_deadline = 0
                if np.random.rand() > RFQ_happen_pro:
                    random_demand = 0
                    random_deadline = 0
                demand_hist.append(random_demand)
                deadline_hist.append(random_deadline)
            demand_hist_list.append(demand_hist)
            deadline_hist_list.append(deadline_hist)
        return demand_hist_list, deadline_hist_list


def build_urgent_rfq_demand_model(seed, parameters):
    if 'demand_test' in parameters and parameters['demand_test'] is not None:
        demand_records = parameters['demand_test']
        forecast = build_teckwah_forecast(demand_records)
        demand_model = UrgentRFQTeckwahDemand(
            seed,
            demand_records,
            forecast,
            parameters['num_retailer'],
            parameters['epi_len'],
        )
    else:
        demand_model = UrgentRFQRandomDemand(
            seed,
            parameters['demand_level'],
            parameters['num_retailer'],
            parameters['epi_len'],
        )
        demand_records = demand_model.gen_demand()
    return demand_model, demand_records, demand_model.gen_urgent_RFQ_demand(parameters['RFQ_happen_pro'])


def build_deadline_rfq_demand_model(seed, parameters):
    if 'demand_test' in parameters and parameters['demand_test'] is not None:
        demand_records = parameters['demand_test']
        forecast = build_teckwah_forecast(demand_records)
        demand_model = DeadlineRFQTeckwahDemand(
            seed,
            demand_records,
            forecast,
            parameters['num_retailer'],
            parameters['epi_len'],
        )
    else:
        demand_model = DeadlineRFQRandomDemand(
            seed,
            parameters['demand_level'],
            parameters['RFQ_deadline_level'],
            parameters['num_retailer'],
            parameters['epi_len'],
        )
        demand_records = demand_model.gen_demand()
    urgent_demand_records, urgent_deadline_records = demand_model.gen_urgent_RFQ_demand(parameters['RFQ_happen_pro'])
    return demand_model, demand_records, urgent_demand_records, urgent_deadline_records


class Retailer:
    def __init__(self, demand_records, number, f,
                 ini_inv, holding, lost_sales, L, LT, capacity, fixed_order,
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
        self.demand_list = demand_records
        self.fixed_order_cost = fixed_order[number]
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


class RentalRetailer(Retailer):
    def __init__(self, demand_records, number, f,
                 ini_inv, holding, lost_sales, L, LT, capacity, production_capacity, fixed_order,
                 per_trans_item, per_trans_order):
        super().__init__(
            demand_records,
            number,
            f,
            ini_inv,
            holding,
            lost_sales,
            L,
            LT,
            capacity,
            fixed_order,
            per_trans_item,
            per_trans_order,
        )
        self.production_capacity = production_capacity[number]

    def order_arrival(self, demand, rental_available):
        self.inv_level = min(self.capacity, self.inv_level + self.pipeline[0])
        self.inv_level -= demand
        if self.inv_level < 0:
            shortage = np.absolute(self.inv_level)
            if shortage <= rental_available:
                self.inv_level = 0
                rental_available = rental_available - shortage
            else:
                self.inv_level = self.inv_level + rental_available
                rental_available = 0
        if rental_available < 0:
            print("Error! rental_available should not be smaller than 0!!!")
        self.pipeline = np.concatenate((self.pipeline[1:], [self.action]))
        return rental_available