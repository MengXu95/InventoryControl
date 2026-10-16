"""Central offline simulator; policies interact only through local Agent interfaces."""

import copy
from dataclasses import asdict

import numpy as np

from .policy import Policy
from .protocol import negotiate, topology_edges
from .scenario import Settings, build_streams, validate_parameters
from .skills import Agent, EPS, Observation

SERVICE_CONSTRAINT = "max_episode_sum_site_fill_shortfall_v1"


def execute_promises(inventory, capacities, promises, unit_cost, fixed_cost):
    """Signed post-demand stocks; priority = (round, donor ID, receiver ID).

    Shared physical ledgers are debited on every execution. A receiver can first
    clear an unmet-demand deficit and then store up to capacity. No re-matching.
    """
    free = [max(0.0, stock) for stock in inventory]
    room = [max(0.0, cap - stock) for stock, cap in zip(inventory, capacities)]
    records, cost = [], 0.0
    for promise in sorted(promises, key=lambda p: (p.round, p.donor, p.receiver)):
        i, j, planned = promise.donor, promise.receiver, promise.quantity
        if planned <= EPS or i == j:
            raise ValueError("Invalid approved promise")
        quantity = min(planned, free[i], room[j])
        if quantity <= EPS:
            quantity = 0.0
        free[i] -= quantity
        room[j] -= quantity
        inventory[i] -= quantity
        inventory[j] += quantity
        status = ("cancelled" if quantity == 0 else
                  "partial" if quantity < planned - EPS else "full")
        if quantity > 0:
            cost += unit_cost * quantity + fixed_cost
        records.append({**asdict(promise), "planned": planned, "executed": quantity,
                        "unexecuted": planned - quantity, "status": status})
    return records, cost


def transfer_metrics(records):
    return {
        "approved_count": len(records),
        "approved_qty": sum(r["planned"] for r in records),
        "executed_count": sum(int(r["executed"] > 0) for r in records),
        "executed_qty": sum(r["executed"] for r in records),
        "cancelled_count": sum(r["status"] == "cancelled" for r in records),
        # Includes the unfulfilled part of partial executions.
        "cancelled_qty": sum(r["unexecuted"] for r in records),
        "partial_count": sum(r["status"] == "partial" for r in records),
        "partial_qty": sum(r["executed"] for r in records if r["status"] == "partial"),
        "partial_shortfall_qty": sum(r["unexecuted"] for r in records
                                     if r["status"] == "partial"),
    }


class Environment:
    def __init__(self, parameters, seed, settings=None, *, streams=None):
        self.parameters = copy.deepcopy(parameters)
        validate_parameters(self.parameters)
        self.settings = settings or Settings()
        self.seed = seed
        self.n = parameters["num_retailer"]
        self.edges = topology_edges(self.n, self.settings.topology)
        self.horizon = self.settings.periods or parameters["epi_len"]
        if self.horizon > parameters["epi_len"]:
            raise ValueError("Requested horizon exceeds benchmark episode")
        actual, means = (build_streams(self.parameters, seed, self.settings)
                         if streams is None else streams)
        self._actual = np.array(actual, dtype=float, copy=True)
        self._means = np.array(means, dtype=float, copy=True)
        if (self._actual.shape != (self.n, self.horizon)
                or self._means.shape != (self.n, self.horizon, parameters["L"])):
            raise ValueError("Demand/forecast stream shape mismatch")
        if (not np.all(np.isfinite(self._actual)) or np.any(self._actual < 0)
                or not np.all(np.isfinite(self._means)) or np.any(self._means < 0)):
            raise ValueError("Demand and forecast streams must be finite and nonnegative")
        # Streams are environment-private; agent observations contain only means.
        self._actual.setflags(write=False)
        self._means.setflags(write=False)

    def observation(self, site, period):
        p = self.parameters
        return Observation(
            float(self.inventory[site]), tuple(self.pipeline[site]),
            tuple(map(float, self._means[site, period])), float(p["capacity"][site]),
            float(p["holding"][site]), float(p["lost_sales"][site]),
            float(p["fixed_order"][site]), float(p["per_unit_order"][site]),
            float(p["per_trans_item"]), float(p["per_trans_order"]),
        )

    def run(self, individual, trace=False):
        policy = Policy(individual)
        p = self.parameters
        self.agents = [Agent(site, policy) for site in range(self.n)]
        self.inventory = list(map(float, p["ini_inv"]))
        # Every outstanding order, INCLUDING the most recently chosen one.
        # Slot 0 arrives now; Q_t appended after period t arrives in t+LT.
        self.pipeline = [[0.0] * p["LT"] for _ in range(self.n)]
        totals = {"order_cost": 0.0, "holding_cost": 0.0, "lost_sales_cost": 0.0,
                  "transfer_cost": 0.0, "demand_qty": 0.0, "lost_qty": 0.0,
                  "arrival_discard_qty": 0.0}
        site_demand = [0.0] * self.n
        site_lost = [0.0] * self.n
        all_records, traces = [], []
        for period in range(self.horizon):
            observations = [self.observation(site, period) for site in range(self.n)]
            for agent, observation in zip(self.agents, observations):
                agent.begin(observation)
            orders = [agent.order for agent in self.agents]
            promises, rounds = negotiate(self.agents, self.edges, self.settings.rounds)
            # Promise stage is finished. Only the environment now reads actuals.
            demand = self._actual[:, period]
            arrivals = [pipeline[0] for pipeline in self.pipeline]
            for site in range(self.n):
                before_cap = self.inventory[site] + arrivals[site]
                totals["arrival_discard_qty"] += max(0.0, before_cap - p["capacity"][site])
                self.inventory[site] = min(p["capacity"][site], before_cap) - float(demand[site])
            records, trans_cost = execute_promises(
                self.inventory, p["capacity"], promises,
                p["per_trans_item"], p["per_trans_order"])
            totals["transfer_cost"] += trans_cost
            losses = [max(0.0, -stock) for stock in self.inventory]
            for site, agent in enumerate(self.agents):
                # Reset per-period history summaries even when nothing executed.
                agent.memory.last_in = agent.memory.last_out = 0.0
                agent.memory.last_lost = losses[site]
                site_demand[site] += float(demand[site])
                site_lost[site] += losses[site]
                self.inventory[site] = max(0.0, self.inventory[site])
                if self.inventory[site] > p["capacity"][site] + EPS:
                    raise AssertionError("Receiver capacity violated")
                totals["holding_cost"] += self.inventory[site] * p["holding"][site]
                totals["lost_sales_cost"] += losses[site] * p["lost_sales"][site]
                totals["order_cost"] += (orders[site] * p["per_unit_order"][site]
                                         + (p["fixed_order"][site] if orders[site] > 0 else 0))
                self.pipeline[site] = self.pipeline[site][1:] + [orders[site]]
            for record in records:
                if record["executed"] > 0:
                    donor, receiver = self.agents[record["donor"]], self.agents[record["receiver"]]
                    donor.memory.last_out += record["executed"]
                    receiver.memory.last_in += record["executed"]
                    event = {"period": period + 1, **record}
                    donor.memory.execution_history.append(dict(event))
                    receiver.memory.execution_history.append(dict(event))
            all_records.extend(records)
            totals["demand_qty"] += float(sum(demand))
            totals["lost_qty"] += sum(losses)
            if trace:
                traces.append({
                    "period": period + 1,
                    "observations": [asdict(o) for o in observations],
                    "orders": orders, "rounds": rounds,
                    "promises": [asdict(promise) for promise in promises],
                    "arrivals": arrivals, "current_demand": demand.tolist(),
                    "executions": records, "lost": losses,
                    "ending_inventory": list(self.inventory),
                    "ending_pipeline": copy.deepcopy(self.pipeline),
                })
        total_cost = sum(totals[key] for key in
                         ("order_cost", "holding_cost", "lost_sales_cost", "transfer_cost"))
        network_fill_rate = (1.0 - totals["lost_qty"] / totals["demand_qty"]
                             if totals["demand_qty"] else 1.0)
        site_metrics = []
        for site in range(self.n):
            fill_rate = 1.0 - site_lost[site] / site_demand[site] if site_demand[site] else 1.0
            site_metrics.append({
                "site": site, "demand_qty": site_demand[site], "lost_qty": site_lost[site],
                "fill_rate": fill_rate,
                "service_violation": max(0.0, self.settings.service_target - fill_rate),
            })
        metrics = {
            **totals, **transfer_metrics(all_records), "total_cost": total_cost,
            "mean_period_cost": total_cost / self.horizon, "periods": self.horizon,
            "network_fill_rate": network_fill_rate,  # Diagnostic, never a service constraint.
            "site_metrics": site_metrics,
            "service_violation": sum(site["service_violation"] for site in site_metrics),
            "service_constraint": SERVICE_CONSTRAINT,
            "service_target": self.settings.service_target,
            "numerical_failures": 0,  # Any nonfinite arithmetic raises, never falls back.
            "seed": self.seed, "stream_seed": self.seed + self.settings.demand_seed_offset,
            "oracle_forecasts": p.get("demand_test") is not None,
        }
        return metrics, traces
