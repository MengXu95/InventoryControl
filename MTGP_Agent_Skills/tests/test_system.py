import copy
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from MTGP_Agent_Skills.__main__ import main, parser
from MTGP_Agent_Skills.environment import Environment, execute_promises, transfer_metrics
from MTGP_Agent_Skills.policy import (
    MAX_HEIGHT, MAX_NODES, ROLES, TERMINALS, Individual, NumericalPolicyError,
    Policy, canonical_json, crossover, div, load_policy, mul, mutate,
    policy_document, random_individual, save_policy, seeded_random, template,
)
from MTGP_Agent_Skills.protocol import (
    Promise, best_edge, candidate_grid, negotiate, phi, topology_edges,
)
from MTGP_Agent_Skills.scenario import (
    Settings, build_streams, parameters_for, preserved_numpy_random,
)
from MTGP_Agent_Skills.skills import Agent, Observation
from MTGP_Agent_Skills.training import (
    EXPERIMENT_SCHEMA, SERVICE_CONSTRAINT, Search, evaluate_batch,
    fitness, run_training, validate_splits,
)
from Utils.inventory_core import build_demand_model


def observation(inventory, forecasts, capacity=100, holding=1, lost=10,
                pipeline=(0, 0), transfer_unit=0, transfer_fixed=0):
    return Observation(inventory, pipeline, tuple(forecasts), capacity,
                       holding, lost, 0, 0, transfer_unit, transfer_fixed)


def agents_for(observations):
    agents = [Agent(i, Policy(template())) for i in range(len(observations))]
    for agent, obs in zip(agents, observations):
        agent.begin(obs)
    return agents


def parameters(sites=2, periods=3, lead=2):
    p = parameters_for("sN2h_1_5b2", 0)
    p.update(num_retailer=sites, epi_len=periods, LT=lead)
    for name, value in (("ini_inv", 0), ("holding", 1), ("lost_sales", 10),
                        ("capacity", 100), ("fixed_order", 0), ("per_unit_order", 0),
                        ("production_capacity", 1)):
        p[name] = [value] * sites
    p.update(per_trans_item=0, per_trans_order=0)
    return p


class ProtocolTests(unittest.TestCase):
    def test_proxy_grid_and_symmetric_edge(self):
        agents = agents_for([observation(10, [0, 0]),
                             observation(0, [5, 5], holding=5)])
        left, right = [agent.advertise() for agent in agents]
        edge = best_edge(left, right, 1, 1)
        self.assertEqual(edge, best_edge(right, left, 1, 1))
        q = edge.quantity
        a = phi(10 - q, 1, 10) - phi(10, 1, 10)
        b = phi(-10, 5, 10) - phi(-10 + q, 5, 10)
        self.assertAlmostEqual(edge.gain, b - a - q - 1)
        self.assertAlmostEqual(edge.gain, (left.holding + right.lost_sales - 1) * q - 1)
        self.assertEqual(q, 10)
        self.assertIsNone(best_edge(left, right, 100, 100))
        grid = candidate_grid(1e-7)
        self.assertEqual(len(grid), len(set(grid)))
        self.assertTrue(all(0 <= q <= 1e-7 for q in grid))
        self.assertEqual(candidate_grid(1.23456789),
                         [0.0, 0.308642, 0.617284, 0.925926, 1.23456789])

    def test_competing_receivers_shared_reservations_and_mutual_tie(self):
        agents = agents_for([observation(10, [1, 1]),
                             observation(0, [3, 3]), observation(0, [3, 3])])
        promises, trace = negotiate(agents, topology_edges(3), 4)
        self.assertEqual((promises[0].donor, promises[0].receiver), (0, 1))
        self.assertAlmostEqual(sum(p.quantity for p in promises), 8)
        self.assertEqual(len(promises), 2)
        self.assertAlmostEqual(agents[0].reserved_out, 8)
        self.assertAlmostEqual(trace[1]["messages"][0]["rho"], 2)
        self.assertAlmostEqual(trace[1]["messages"][0]["remaining"], 2)
        for row in trace:
            used = []
            for p in row["confirmed"]:
                used.extend([p["donor"], p["receiver"]])
                self.assertEqual(row["choices"][p["donor"]], row["choices"][p["receiver"]])
            self.assertEqual(len(used), len(set(used)))
        with self.assertRaises(ValueError):
            agents[0].reserve(1)

    def test_reservations_update_messages_not_frozen_gp_features(self):
        class RecordingPolicy(Policy):
            def __init__(self):
                super().__init__(template())
                self.calls = []

            def output(self, role, features):
                self.calls.append((role, dict(features)))
                return super().output(role, features)

        policy = RecordingPolicy()
        observations = [observation(10, [1, 1]), observation(0, [3, 3]),
                        observation(0, [3, 3])]
        agents = [Agent(site, policy) for site in range(3)]
        for agent, obs in zip(agents, observations):
            agent.begin(obs)
        features_before = [agent.features() for agent in agents]
        calls_before = list(policy.calls)
        budgets_before = [agent.quantity for agent in agents]
        self.assertEqual([role for role, _ in calls_before],
                         ["order", "offer", "order", "request", "order", "request"])
        promises, _ = negotiate(agents, topology_edges(3), 3)
        self.assertEqual(len(promises), 2)
        self.assertEqual(policy.calls, calls_before)  # no per-round GP re-evaluation
        self.assertEqual([agent.features() for agent in agents], features_before)
        self.assertEqual([agent.quantity for agent in agents], budgets_before)
        self.assertEqual(agents[0].features()["RHO"], 8)
        self.assertEqual(agents[0].advertise().rho, 0)
        self.assertEqual(agents[0].advertise().remaining, 0)

    def test_topology_restriction_and_generic_five_sites(self):
        observations = [observation(10, [0, 0]), observation(0, [0, 0]),
                        observation(0, [5, 5]), observation(0, [0, 0]),
                        observation(0, [0, 0])]
        agents = agents_for(observations)
        edges = topology_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(negotiate(agents, edges, 3)[0], [])
        agents = agents_for(observations)
        promises, _ = negotiate(agents, topology_edges(5), 3)
        self.assertEqual([(p.donor, p.receiver) for p in promises], [(0, 2)])
        self.assertEqual(len(topology_edges(5)), 10)
        with self.assertRaises(ValueError):
            topology_edges(5, [(0, 5)])

    def test_skill_bounds_and_no_production_cap(self):
        huge = Policy(Individual(["1000000.0", "1000000.0", "1000000.0"]))
        receiver = Agent(0, huge)
        receiver.begin(observation(9, [10, 10], capacity=10))
        self.assertEqual(receiver.quantity, 1)
        self.assertEqual(receiver.order, 10)
        donor = Agent(1, huge)
        donor.begin(observation(1, [0, 0], pipeline=(2, 30)))
        self.assertEqual(donor.quantity, 3)  # not all future pipeline is available
        negative = Agent(2, Policy(Individual(["sub(0.0, 1.0)"] * 3)))
        negative.begin(observation(10, [0, 0]))
        self.assertEqual((negative.order, negative.quantity), (0, 0))
        idle = agents_for([observation(4, [2, 2])])[0]
        self.assertEqual(idle.role, "idle")

    def test_receiver_capacity_partial_and_cancelled_execution(self):
        inventory = [5.0, 9.0, -3.0, 0.0]
        promises = [Promise(0, 0, 1, 4, 1), Promise(1, 0, 2, 6, 1),
                    Promise(2, 0, 3, 2, 1)]
        records, cost = execute_promises(inventory, [10] * 4, promises, 2, 3)
        self.assertEqual([r["executed"] for r in records], [1, 4, 0])
        self.assertEqual([r["status"] for r in records], ["partial", "partial", "cancelled"])
        self.assertEqual(inventory, [0, 10, 1, 0])
        self.assertEqual(cost, 16)  # 2*5 + 3*2; zero transfer pays nothing
        metrics = transfer_metrics(records)
        self.assertEqual(metrics["approved_qty"], 12)
        self.assertEqual(metrics["executed_qty"], 5)
        self.assertEqual(metrics["cancelled_qty"], 7)
        self.assertEqual(metrics["partial_count"], 2)
        self.assertEqual(metrics["cancelled_count"], 1)


class EnvironmentTests(unittest.TestCase):
    def test_pre_demand_observation_and_no_actual_leakage(self):
        p = parameters(periods=2)
        p["ini_inv"] = [10, 0]
        means = np.array([[[0, 0], [0, 0]], [[5, 5], [5, 5]]], dtype=float)
        demand1 = np.zeros((2, 2))
        demand2 = np.array([[100, 200], [100, 300]], dtype=float)
        env1 = Environment(p, 0, streams=(demand1, means))
        env2 = Environment(p, 0, streams=(demand2, means))
        _, a = env1.run(template(), trace=True)
        _, b = env2.run(template(), trace=True)
        for key in ("observations", "orders", "rounds", "promises"):
            self.assertEqual(a[0][key], b[0][key])
        self.assertNotIn("demand", env1.agents[0].features())
        self.assertFalse(any("ID" in term or "GLOBAL" in term for terms in TERMINALS.values() for term in terms))
        self.assertFalse(hasattr(env1.agents[0], "_actual"))
        self.assertTrue(a[0]["promises"])
        self.assertEqual(b[0]["executions"][0]["executed"], 0)

    def test_partial_actual_history_and_losses_reset(self):
        p = parameters(sites=3, periods=2)
        p["ini_inv"] = [10, 0, 0]
        p["fixed_order"] = [0] * 3
        means = np.array([[[1, 1], [0, 0]], [[3, 3], [0, 0]],
                          [[3, 3], [0, 0]]], dtype=float)
        actual = np.array([[5, 0], [0, 0], [0, 0]], dtype=float)
        env = Environment(p, 0, streams=(actual, means))
        metrics, trace = env.run(Individual(["0.0", "DEFICIT", "SURPLUS"]), trace=True)
        self.assertEqual(metrics["approved_qty"], 8)
        self.assertEqual(metrics["executed_qty"], 5)
        self.assertEqual(metrics["partial_count"], 1)
        self.assertEqual(metrics["cancelled_count"], 1)
        json.dumps(metrics, allow_nan=False)  # public metrics must be JSON-native
        self.assertEqual(len(env.agents[0].memory.execution_history), 1)
        self.assertEqual(env.agents[0].memory.execution_history[0]["executed"], 5)
        self.assertEqual(env.agents[0].memory.last_out, 0)  # second-period reset
        self.assertEqual(trace[0]["ending_inventory"], [0, 5, 0])
        p = parameters(periods=2)
        env = Environment(p, 0, streams=(np.array([[5, 0], [0, 0]]), np.zeros((2, 2, 2))))
        metrics, trace = env.run(Individual(["0.0"] * 3), trace=True)
        self.assertEqual([row["lost"][0] for row in trace], [5, 0])
        self.assertEqual(env.agents[0].memory.last_lost, 0)
        self.assertEqual(metrics["lost_sales_cost"], 50)

    def test_pipeline_lead_time_all_outstanding_and_elapsed_denominator(self):
        for lead in (1, 2, 3):
            p = parameters(periods=5, lead=lead)
            env = Environment(p, 0, streams=(np.zeros((2, 5)), np.zeros((2, 5, 2))))
            metrics, trace = env.run(Individual(["7.0", "0.0", "0.0"]), trace=True)
            self.assertEqual([row["arrivals"][0] for row in trace],
                             [0] * lead + [7] * (5 - lead))
            self.assertEqual(trace[1]["observations"][0]["pipeline"][-1], 7)
            self.assertEqual(sum(trace[min(lead, 4)]["observations"][0]["pipeline"]), 7 * lead)
            self.assertEqual(metrics["periods"], 5)
            self.assertEqual(metrics["mean_period_cost"], metrics["total_cost"] / 5)
        p = parameters_for("sN2h_1_5b2", 0)
        metrics, _ = Environment(p, 7).run(template())
        self.assertEqual(metrics["periods"], 64)
        metrics, _ = Environment(p, 7, Settings(periods=3)).run(template())
        self.assertEqual(metrics["periods"], 3)

    def test_benchmark_exact_stream_reset_and_rng_preservation(self):
        p = parameters_for("sN2h_1_5b2", 0)
        np.random.seed(123)
        saved = np.random.get_state()
        actual, forecasts = build_streams(p, 7, Settings())
        after = np.random.get_state()
        self.assertTrue(np.array_equal(saved[1], after[1]))
        self.assertEqual(saved[2:], after[2:])
        model, _ = build_demand_model(7, p)
        model.reset()
        expected = np.array(model.gen_demand())[:, :64]
        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(forecasts[0, 0, 0], model.f(0, 1))
        self.assertEqual(forecasts[1, 63, 1], model.f(1, 65))
        prefix, _ = build_streams(p, 7, Settings(periods=3))
        self.assertTrue(np.array_equal(prefix, actual[:, :3]))

    def test_scenario_matched_64_step_demand_arrival_and_cost_trajectory(self):
        """Differential check of benchmark staging, not its old GP action map.

        Use the actual shared scenario without modifying LT, L, costs or horizon.
        A varying local-mean order tree supplies the SAME order sequence to both
        the new environment and an independent previous-action staging reference.
        No original algorithm package is imported or executed.
        """
        scenario = "sN2h_1_5b2"
        settings = Settings(rounds=0)
        individual = Individual(["add(FC1, FC2)", "0.0", "0.0"])
        splits = {"train": (11, 12), "validation": (1001, 1002), "test": (2001, 2002)}
        for split, seeds in splits.items():
            for seed in seeds:
                with self.subTest(split=split, seed=seed):
                    p = parameters_for(scenario, seed)
                    self.assertEqual((p["epi_len"], p["LT"], p["L"]), (64, 2, 2))
                    with preserved_numpy_random():
                        model, _ = build_demand_model(seed, p)
                        model.reset()  # exact old run() reset, NOT a different RNG
                        demand_records = model.gen_demand()
                    self.assertEqual(len(demand_records[0]), 65)
                    metrics, events = Environment(p, seed, settings).run(individual, trace=True)
                    self.assertEqual(len(events), 64)
                    self.assertEqual([row["period"] for row in events], list(range(1, 65)))
                    self.assertEqual(metrics["seed"], seed)
                    self.assertEqual(metrics["stream_seed"], seed + 1000)
                    stock = list(map(float, p["ini_inv"]))
                    # Old code keeps the latest action OUTSIDE its LT-1 pipeline.
                    old_pipeline = [[0.0] * (p["LT"] - 1) for _ in stock]
                    previous_action = [0.0] * len(stock)
                    costs = {"order_cost": 0.0, "holding_cost": 0.0, "lost_sales_cost": 0.0}
                    discarded = 0.0
                    lost_total = 0.0
                    for index, row in enumerate(events):
                        period = index + 1
                        self.assertEqual(row["current_demand"],
                                         [float(records[index]) for records in demand_records])
                        self.assertEqual(row["arrivals"], [pipe[0] for pipe in old_pipeline])
                        self.assertEqual(row["promises"], [])
                        self.assertEqual(row["executions"], [])
                        for site in range(p["num_retailer"]):
                            obs = row["observations"][site]
                            forecast = [float(model.f(site, period + k)) for k in range(p["L"])]
                            self.assertEqual(obs["inventory"], stock[site])
                            self.assertEqual(obs["forecasts"], tuple(forecast))
                            self.assertEqual(obs["pipeline"],
                                             tuple(old_pipeline[site] + [previous_action[site]]))
                            order = min(p["capacity"][site], max(0.0, forecast[0] + forecast[1]))
                            self.assertEqual(row["orders"][site], order)
                            before_cap = stock[site] + old_pipeline[site][0]
                            discarded += max(0.0, before_cap - p["capacity"][site])
                            signed = min(p["capacity"][site], before_cap) - float(demand_records[site][index])
                            loss = max(0.0, -signed)
                            stock[site] = max(0.0, signed)
                            self.assertEqual(row["lost"][site], loss)
                            self.assertEqual(row["ending_inventory"][site], stock[site])
                            costs["order_cost"] += (order * p["per_unit_order"][site]
                                                    + (p["fixed_order"][site] if order > 0 else 0))
                            costs["holding_cost"] += stock[site] * p["holding"][site]
                            costs["lost_sales_cost"] += loss * p["lost_sales"][site]
                            lost_total += loss
                            old_pipeline[site] = old_pipeline[site][1:] + [previous_action[site]]
                            previous_action[site] = order
                            self.assertEqual(row["ending_pipeline"][site],
                                             old_pipeline[site] + [previous_action[site]])
                    for name, value in costs.items():
                        self.assertEqual(metrics[name], value)
                    self.assertEqual(metrics["arrival_discard_qty"], discarded)
                    self.assertEqual(metrics["lost_qty"], lost_total)
                    self.assertEqual(metrics["demand_qty"],
                                     sum(sum(records[:64]) for records in demand_records))
                    self.assertEqual(metrics["periods"], 64)
                    self.assertEqual(metrics["total_cost"], sum(costs.values()))
                    self.assertEqual(metrics["mean_period_cost"], sum(costs.values()) / 64)
                    for site, record in enumerate(metrics["site_metrics"]):
                        self.assertEqual(record["site"], site)
                        self.assertEqual(record["demand_qty"], sum(demand_records[site][:64]))
                        self.assertEqual(record["lost_qty"], sum(row["lost"][site] for row in events))
                        expected_fill = 1 - record["lost_qty"] / record["demand_qty"]
                        self.assertEqual(record["fill_rate"], expected_fill)
                        self.assertEqual(record["service_violation"], max(0, .95 - expected_fill))
                    self.assertEqual(metrics["service_violation"],
                                     sum(site["service_violation"] for site in metrics["site_metrics"]))
                    self.assertGreater(len(set(row["orders"][0] for row in events)), 60)
                    # Q_1 arrives in period 3; Q_62 is the final in-horizon arrival.
                    self.assertEqual(events[2]["arrivals"], events[0]["orders"])
                    self.assertEqual(events[63]["arrivals"], events[61]["orders"])
                    self.assertEqual(events[63]["ending_pipeline"],
                                     [list(q) for q in zip(events[62]["orders"], events[63]["orders"])])

    def test_historical_requires_explicit_oracle_and_no_padding(self):
        p = parameters(periods=3)
        p["demand_test"] = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "future actuals"):
            Environment(p, 0)
        env = Environment(p, 0, Settings(allow_oracle_forecasts=True))
        metrics, _ = env.run(template())
        self.assertTrue(metrics["oracle_forecasts"])
        p["demand_test"] = np.array([[1, 2, 3], [2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "Insufficient forecast"):
            Environment(p, 0, Settings(allow_oracle_forecasts=True))

    def test_generic_five_site_episode(self):
        p = parameters_for("sN5h_1_2_3_4_5b2", 0)
        env = Environment(p, 11, Settings(periods=8, topology=((0, 1), (1, 2), (2, 3), (3, 4))))
        metrics, trace = env.run(template(), trace=True)
        self.assertEqual(len(env.agents), 5)
        self.assertEqual(metrics["periods"], 8)
        for row in trace:
            for promise in row["promises"]:
                self.assertIn(tuple(sorted((promise["donor"], promise["receiver"]))), env.edges)
            self.assertTrue(all(0 <= stock <= 100 for stock in row["ending_inventory"]))


class ServiceConstraintTests(unittest.TestCase):
    def test_network_fill_cannot_hide_a_neglected_site(self):
        p = parameters(periods=1)
        p["ini_inv"] = [100, 0]
        env = Environment(p, 11, Settings(rounds=0),
                          streams=(np.array([[100], [1]]), np.zeros((2, 1, 2))))
        metrics, _ = env.run(Individual(["0.0"] * 3))
        # Network fill is 100/101 > .95, but the second site has fill 0.
        self.assertAlmostEqual(metrics["service_violation"], .95)
        self.assertGreaterEqual(metrics["network_fill_rate"], .95)
        self.assertEqual(metrics["site_metrics"], [
            {"site": 0, "demand_qty": 100.0, "lost_qty": 0.0,
             "fill_rate": 1.0, "service_violation": 0.0},
            {"site": 1, "demand_qty": 1.0, "lost_qty": 1.0,
             "fill_rate": 0.0, "service_violation": .95},
        ])

    def test_zero_demand_sites_have_fill_one_and_no_violation(self):
        p = parameters(sites=3, periods=2)
        means = np.zeros((3, 2, 2))
        env = Environment(p, 11, Settings(rounds=0),
                          streams=(np.array([[0, 0], [2, 3], [0, 0]]), means))
        metrics, _ = env.run(Individual(["0.0"] * 3))
        self.assertEqual([site["fill_rate"] for site in metrics["site_metrics"]], [1, 0, 1])
        self.assertEqual([site["service_violation"] for site in metrics["site_metrics"]], [0, .95, 0])
        self.assertEqual(metrics["service_violation"], .95)
        env = Environment(p, 12, Settings(rounds=0),
                          streams=(np.zeros((3, 2)), means))
        metrics, _ = env.run(Individual(["0.0"] * 3))
        self.assertEqual(metrics["network_fill_rate"], 1)
        self.assertEqual(metrics["service_violation"], 0)
        self.assertTrue(all(site["fill_rate"] == 1 for site in metrics["site_metrics"]))

    def test_batch_constraint_is_max_episode_sum_site_shortfalls(self):
        p = parameters(periods=1)
        p["ini_inv"] = [5, 5]
        settings = Settings(rounds=0, periods=1)
        episodes = [
            Environment(p, seed, settings,
                        streams=(np.array(actual), np.zeros((2, 1, 2))))
            for seed, actual in ((11, [[10], [10]]), (12, [[20], [5]]))
        ]
        with patch("MTGP_Agent_Skills.training.Environment", side_effect=episodes):
            summary, _ = evaluate_batch(Individual(["0.0"] * 3), "sN2h_1_5b2",
                                        (11, 12), settings)
        # Episode A: .45+.45=.90; B: .70+0=.70. Required V=.90.
        # Neither mean episode violation (.80), max site shortfall (.70), nor
        # sum of per-site episode maxima (1.15) is the required constraint.
        self.assertAlmostEqual(summary["worst_service_violation"], .9)
        self.assertAlmostEqual(summary["episodes"][0]["service_violation"], .9)
        self.assertAlmostEqual(summary["episodes"][1]["service_violation"], .7)
        self.assertAlmostEqual(summary["mean_network_fill_rate"], .45)
        self.assertEqual(summary["mean_site_fill_rates"], [.375, .75])
        self.assertEqual(summary["mean_period_cost"],
                         sum(e["total_cost"] / e["periods"] for e in summary["episodes"]) / 2)
        self.assertEqual(fitness(summary),
                         (summary["worst_service_violation"], summary["mean_period_cost"], 3))


class PolicyAndSearchTests(unittest.TestCase):
    def test_three_trees_serialization_and_reject_unknown_code(self):
        individual = template()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.json"
            save_policy(individual, path)
            restored = load_policy(path)
            self.assertEqual(policy_document(restored), policy_document(individual))
            self.assertEqual(path.read_text(), canonical_json(policy_document(restored)))
        for expressions in (["I", "F"], ["ROOM", "F", "I"],
                            ["__import__('os')", "F", "I"], ["1e999", "F", "I"],
                            ["I + F", "F", "I"], ["CAPACITY_OTHER", "F", "I"]):
            with self.assertRaises(ValueError):
                Individual(expressions)
        self.assertEqual(div(1, 0), 1)
        self.assertEqual(mul(1e6, 1e6), 1e6)

    def test_numerical_failures_are_explicit(self):
        class BadPolicy:
            def output(self, role, features):
                return math.nan
        with self.assertRaises(NumericalPolicyError):
            Agent(0, BadPolicy()).begin(observation(0, [2, 2]))
        policy = Policy(template())
        agent = agents_for([observation(0, [2, 2])])[0]
        features = agent.features()
        features["I"] = math.inf
        with self.assertRaises(NumericalPolicyError):
            policy.output("order", features)

    def test_module_aware_operators_bloat_and_seed_repeatability(self):
        def exercise():
            left = Individual(["ORDER_FIXED", "ROOM", "AVAIL"])
            right = Individual(["ORDER_UNIT", "DEFICIT", "SURPLUS"])
            signatures = []
            for _ in range(40):
                first, second = crossover(left, right)
                self.assertLessEqual(sum(str(a) != str(b) for a, b in zip(first, left)), 1)
                child = mutate(first)
                self.assertLessEqual(sum(str(a) != str(b) for a, b in zip(first, child)), 1)
                for role, tree in zip(ROLES, child):
                    self.assertLessEqual(tree.height, MAX_HEIGHT)
                    self.assertLessEqual(len(tree), MAX_NODES)
                    # Reconstruction verifies role terminals.
                signatures.append(child.expressions())
                left, right = child, second
            return signatures
        before = random.getstate()
        with seeded_random(13):
            first = exercise()
        self.assertEqual(before, random.getstate())
        with seeded_random(13):
            second = exercise()
        self.assertEqual(first, second)

    def test_splits_and_constraint_first_fitness(self):
        with self.assertRaises(ValueError):
            validate_splits((1,), (1,), (2,))
        with self.assertRaises(ValueError):
            validate_splits((1, 1), (2,), (3,))
        self.assertLess(fitness({"worst_service_violation": 0, "mean_period_cost": 1e9, "size": 100}),
                        fitness({"worst_service_violation": .01, "mean_period_cost": 0, "size": 1}))

    def test_batch_aggregation_uses_worst_violation_and_equal_episode_weights(self):
        summary, _ = evaluate_batch(Individual(["0.0"] * 3), "sN2h_1_5b2",
                                    (2001, 2002), Settings())
        episodes = summary["episodes"]
        self.assertEqual([episode["seed"] for episode in episodes], [2001, 2002])
        self.assertEqual([episode["periods"] for episode in episodes], [64, 64])
        self.assertEqual(summary["worst_service_violation"],
                         max(episode["service_violation"] for episode in episodes))
        self.assertEqual(summary["mean_period_cost"],
                         sum(episode["total_cost"] / 64 for episode in episodes) / 2)
        self.assertEqual(summary["mean_network_fill_rate"],
                         sum(episode["network_fill_rate"] for episode in episodes) / 2)
        pooled_fill = 1 - sum(e["lost_qty"] for e in episodes) / sum(e["demand_qty"] for e in episodes)
        self.assertNotAlmostEqual(summary["mean_network_fill_rate"], pooled_fill)
        for episode in episodes:
            self.assertEqual(episode["service_violation"],
                             sum(site["service_violation"] for site in episode["site_metrics"]))
        self.assertEqual(summary["size"], 3)

    def test_validation_selection_precedes_any_heldout_evaluation(self):
        scenario = "sN2h_1_5b2"
        train, validation, test = (11, 12), (1001, 1002), (2001, 2002)
        evaluated = []
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"

            def recorded_evaluation(individual, name, seeds, settings, trace=False):
                seeds = tuple(seeds)
                self.assertEqual(name, scenario)
                self.assertIn(seeds, (train, validation, test))
                if seeds == test:
                    self.assertIn(validation, evaluated)
                    # Selection is committed to disk before ANY test outcome exists.
                    selected = load_policy(output / "best_policy.json")
                    self.assertEqual(selected.expressions(), individual.expressions())
                evaluated.append(seeds)
                return evaluate_batch(individual, name, seeds, settings, trace=trace)

            with patch("MTGP_Agent_Skills.training.evaluate_batch",
                       side_effect=recorded_evaluation):
                _, metrics = run_training(scenario, Settings(), Search(population=4, generations=1),
                                           train, validation, test, output, trace=False)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["schema"], EXPERIMENT_SCHEMA)
            self.assertEqual(config["service_constraint"], SERVICE_CONSTRAINT)
            self.assertEqual(config["splits"],
                             {"train": list(train), "validation": list(validation), "test": list(test)})
            self.assertEqual(evaluated[-3:], [train, validation, test])
            self.assertEqual(evaluated.count(test), 1)
            self.assertTrue(all(seeds == train for seeds in evaluated[:evaluated.index(validation)]))
            for split, seeds in (("train", train), ("validation", validation), ("test", test)):
                self.assertEqual([episode["seed"] for episode in metrics[split]["episodes"]], list(seeds))
                self.assertTrue(all(episode["periods"] == 64 for episode in metrics[split]["episodes"]))

    def test_small_training_repeats_and_reload_matches_heldout(self):
        settings = Settings(periods=6)
        search = Search(seed=17, population=4, generations=1)
        with tempfile.TemporaryDirectory() as temp:
            a = Path(temp) / "a"
            b = Path(temp) / "b"
            chosen, metrics = run_training("sN2h_1_5b2", settings, search,
                                           (11,), (1001,), (2001,), a)
            repeated, repeated_metrics = run_training("sN2h_1_5b2", settings, search,
                                                      (11,), (1001,), (2001,), b)
            self.assertEqual(chosen.expressions(), repeated.expressions())
            self.assertEqual(metrics, repeated_metrics)
            for name in ("config.json", "metrics.json", "history.json", "best_policy.json",
                         "training_evaluations.json", "traces_test.json"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            reloaded = load_policy(a / "best_policy.json")
            summary, _ = evaluate_batch(reloaded, "sN2h_1_5b2", (2001,), settings)
            self.assertEqual(summary, metrics["test"])
            evaluation = Path(temp) / "replay"
            stdout = StringIO()
            with redirect_stdout(stdout):
                main(["evaluate", "--policy", str(a / "best_policy.json"), "--config",
                      str(a / "config.json"), "--seeds", "2001", "--output-dir", str(evaluation)])
            self.assertEqual(json.loads(stdout.getvalue())["summary"], metrics["test"])
            self.assertEqual(json.loads((evaluation / "metrics.json").read_text()), metrics["test"])

    def test_cli_defaults(self):
        args = parser().parse_args(["smoke"])
        self.assertEqual(args.population, 6)
        self.assertEqual(args.generations, 1)

    def test_cli_rejects_old_or_incompatible_service_config(self):
        with tempfile.TemporaryDirectory() as temp:
            policy_path = Path(temp) / "policy.json"
            config_path = Path(temp) / "config.json"
            output = Path(temp) / "must_not_be_written"
            save_policy(template(), policy_path)
            for config in ({"schema": "peer-experiment-v1"},
                           {"schema": EXPERIMENT_SCHEMA, "service_constraint": "network_fill"}):
                with self.subTest(config=config):
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "site-level service"):
                        main(["evaluate", "--policy", str(policy_path), "--config",
                              str(config_path), "--output-dir", str(output)])
                    self.assertFalse(output.exists())

    def test_cli_full_horizon_replay_with_real_transfers(self):
        # Regression: NumPy quantities could leak int64 counts into CLI stdout.
        individual = Individual(["maximum(maximum(FC1, RHO), DUE)",
                                 "maximum(F, LAST_OUT)", "maximum(1.0, CAP)"])
        with tempfile.TemporaryDirectory() as temp:
            policy_path = Path(temp) / "policy.json"
            output = Path(temp) / "evaluation"
            save_policy(individual, policy_path)
            stdout = StringIO()
            with redirect_stdout(stdout):
                main(["evaluate", "--policy", str(policy_path), "--seeds", "2001,2002",
                      "--output-dir", str(output), "--no-trace"])
            summary = json.loads((output / "metrics.json").read_text())
            self.assertEqual(json.loads(stdout.getvalue())["summary"], summary)
            self.assertGreater(sum(e["executed_count"] for e in summary["episodes"]), 0)


if __name__ == "__main__":
    unittest.main()
