"""Read-only adapters for benchmark scenarios and exact synthetic demand streams."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np

from Utils.ScenarioDesign_replenish_transship import ScenarioDesign_replenish_transship
from Utils.inventory_core import build_demand_model

from .protocol import topology_edges


@contextmanager
def preserved_numpy_random():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass(frozen=True)
class Settings:
    rounds: int = 3
    periods: int | None = None
    topology: tuple | None = None
    demand_seed_offset: int = 1000
    allow_oracle_forecasts: bool = False
    service_target: float = 0.95

    def __post_init__(self):
        if type(self.rounds) is not int or self.rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if self.periods is not None and (type(self.periods) is not int or self.periods < 1):
            raise ValueError("periods must be a positive integer")
        if not 0 <= self.service_target <= 1:
            raise ValueError("service target must be in [0, 1]")
        if type(self.demand_seed_offset) is not int:
            raise ValueError("demand seed offset must be an integer")

    def document(self):
        return asdict(self)


def parameters_for(name, seed):
    with preserved_numpy_random():
        return ScenarioDesign_replenish_transship(name).get_parameter(seed=seed)


def validate_parameters(parameters):
    p = parameters
    for name in ("num_retailer", "LT", "L", "epi_len"):
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    n = p["num_retailer"]
    topology_edges(n)
    for name in ("ini_inv", "holding", "lost_sales", "capacity",
                 "fixed_order", "per_unit_order"):
        values = np.asarray(p[name], dtype=float)
        if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Invalid site vector: {name}")
    for name in ("per_trans_item", "per_trans_order"):
        if not np.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f"Invalid cost: {name}")
    if any(i > c for i, c in zip(p["ini_inv"], p["capacity"])):
        raise ValueError("Initial inventory exceeds capacity")


def build_streams(parameters, seed, settings):
    """Private environment data, never passed to an Agent.

    Full benchmark episode is generated even when only a prefix is simulated.
    Starting directly at seed+1000 is identical to the old run() reset stream.
    """
    if parameters.get("demand_test") is not None and not settings.allow_oracle_forecasts:
        raise ValueError("Historical demand_test exposes future actuals as forecasts; "
                         "requires explicit --allow-oracle-forecasts (not causal)")
    horizon = settings.periods or parameters["epi_len"]
    if horizon > parameters["epi_len"]:
        raise ValueError("Cannot extend benchmark horizon with --periods")
    with preserved_numpy_random():
        model, demands = build_demand_model(seed + settings.demand_seed_offset, parameters)
        try:
            forecasts = [[[model.f(site, period + k)
                           for k in range(parameters["L"])]
                          for period in range(1, horizon + 1)]
                         for site in range(parameters["num_retailer"])]
        except IndexError as exc:
            raise ValueError("Insufficient forecast records: reduce --periods; "
                             "no future-actual padding is performed") from exc
    actual = np.array(demands, dtype=float)
    means = np.array(forecasts, dtype=float)
    if actual.shape[1] < horizon:
        raise ValueError("Insufficient actual demand records")
    # Match old len(demand_records)-1 termination when records are short.
    if len(actual[0]) <= horizon:
        raise ValueError("Benchmark needs one terminal demand record beyond the horizon")
    return actual[:, :horizon], means
