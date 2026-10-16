"""A site Agent sees only its immutable local observation and own execution memory."""

from dataclasses import dataclass, field

from .policy import finite

EPS = 1e-8


@dataclass(frozen=True)
class Observation:
    inventory: float
    pipeline: tuple
    forecasts: tuple
    capacity: float
    holding: float
    lost_sales: float
    fixed_order: float
    unit_order: float
    transfer_unit: float
    transfer_fixed: float

    @property
    def rho(self):
        return self.inventory + sum(self.pipeline) - sum(self.forecasts)


@dataclass
class Memory:
    last_in: float = 0.0
    last_out: float = 0.0
    last_lost: float = 0.0
    # Only positive ACTUAL transfers; never approved/cancelled promises.
    execution_history: list = field(default_factory=list)


@dataclass(frozen=True)
class Message:
    site: int
    role: str
    remaining: float
    rho: float
    holding: float
    lost_sales: float
    reserved_out: float
    reserved_in: float


class Agent:
    def __init__(self, site, policy):
        self.site = site  # Routing/tie-breaking only, never a tree feature.
        self.policy = policy
        self.memory = Memory()

    def begin(self, observation):
        self.observation = observation
        self.reserved_out = self.reserved_in = 0.0
        features = self.features()
        self.order = min(observation.capacity,
                         max(0.0, finite(self.policy.output("order", features))))
        self.role = ("offer" if observation.rho > EPS else
                     "request" if observation.rho < -EPS else "idle")
        self.quantity = 0.0
        if self.role != "idle":
            bound = (min(features["SURPLUS"], features["AVAIL"])
                     if self.role == "offer" else
                     min(features["DEFICIT"], features["ROOM"]))
            self.quantity = min(bound, max(0.0, finite(
                self.policy.output(self.role, features))))

    def features(self):
        o, m = self.observation, self.memory
        return {
            "I": o.inventory, "PIPE": sum(o.pipeline), "DUE": o.pipeline[0],
            "F": sum(o.forecasts), "FC1": o.forecasts[0],
            "FC2": o.forecasts[1] if len(o.forecasts) > 1 else 0.0,
            "CAP": o.capacity, "H": o.holding, "B": o.lost_sales, "RHO": o.rho,
            "LAST_IN": m.last_in, "LAST_OUT": m.last_out, "LAST_LOST": m.last_lost,
            "ORDER_FIXED": o.fixed_order, "ORDER_UNIT": o.unit_order,
            "ROOM": max(0.0, o.capacity - o.inventory),
            "DEFICIT": max(0.0, -o.rho), "SURPLUS": max(0.0, o.rho),
            "AVAIL": min(o.capacity, o.inventory + o.pipeline[0]),
            "TC": o.transfer_unit, "TK": o.transfer_fixed,
        }

    def advertise(self):
        return Message(
            self.site, self.role, max(0.0, self.quantity - self.reserved_out - self.reserved_in),
            self.observation.rho - self.reserved_out + self.reserved_in,
            self.observation.holding, self.observation.lost_sales,
            self.reserved_out, self.reserved_in,
        )

    def reserve(self, quantity):
        if quantity < -EPS or quantity > self.advertise().remaining + EPS:
            raise ValueError("Reservation exceeds remaining skill budget")
        if self.role == "offer":
            self.reserved_out += quantity
        elif self.role == "request":
            self.reserved_in += quantity
        else:
            raise ValueError("Idle site cannot reserve")

    def best_partner(self, inbox):
        """Local bilateral evaluation only: own message + received neighbors."""
        from .protocol import best_edge
        own = self.advertise()
        candidates = []
        for peer in inbox:
            edge = best_edge(own, peer, self.observation.transfer_unit,
                             self.observation.transfer_fixed)
            if edge is not None:
                candidates.append(edge)
        return min(candidates, key=lambda edge: edge.rank) if candidates else None
