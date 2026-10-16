"""Finite-round mutual-choice protocol. Delivery is simulated, decisions are local."""

from dataclasses import dataclass

from .skills import EPS

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_DECIMALS = 6


def phi(rho, holding, lost_sales):
    return holding * max(rho, 0.0) + lost_sales * max(-rho, 0.0)


@dataclass(frozen=True)
class Edge:
    donor: int
    receiver: int
    quantity: float
    gain: float

    @property
    def rank(self):
        # Same comparison at both endpoints, including ties.
        return (-self.gain, self.donor, self.receiver, -self.quantity)


@dataclass(frozen=True)
class Promise:
    round: int
    donor: int
    receiver: int
    quantity: float
    gain: float


def candidate_grid(limit):
    values = []
    for fraction in GRID:
        q = min(limit, round(fraction * limit, GRID_DECIMALS))
        if not any(abs(q - other) <= EPS for other in values):
            values.append(q)
    return values


def best_edge(first, second, unit_cost, fixed_cost):
    if first.role == "request" and second.role == "offer":
        first, second = second, first
    if first.role != "offer" or second.role != "request":
        return None
    donor, receiver = first, second
    limit = min(donor.remaining, receiver.remaining)
    candidates = []
    for q in candidate_grid(limit):
        if q <= EPS:
            continue
        a = (phi(donor.rho - q, donor.holding, donor.lost_sales)
             - phi(donor.rho, donor.holding, donor.lost_sales))
        b = (phi(receiver.rho, receiver.holding, receiver.lost_sales)
             - phi(receiver.rho + q, receiver.holding, receiver.lost_sales))
        gain = b - a - (unit_cost * q + fixed_cost)
        if gain > EPS:
            candidates.append(Edge(donor.site, receiver.site, q, gain))
    return min(candidates, key=lambda edge: edge.rank) if candidates else None


def topology_edges(sites, edges=None):
    if sites < 2:
        raise ValueError("At least two sites required")
    if edges is None:
        return tuple((i, j) for i in range(sites) for j in range(i + 1, sites))
    normalized = set()
    for i, j in edges:
        if type(i) is not int or type(j) is not int or not (0 <= i < sites and 0 <= j < sites) or i == j:
            raise ValueError("Topology needs distinct valid integer site IDs")
        normalized.add(tuple(sorted((i, j))))
    return tuple(sorted(normalized))


def negotiate(agents, edges, rounds):
    """No global edge ranking/solver: deliver, local choices, mutual confirmation."""
    promises, trace = [], []
    for round_index in range(rounds):
        messages = [agent.advertise() for agent in agents]
        inboxes = [[] for _ in agents]
        for i, j in edges:
            inboxes[i].append(messages[j])
            inboxes[j].append(messages[i])
        choices = [agent.best_partner(inbox)
                   for agent, inbox in zip(agents, inboxes)]
        confirmed = []
        for site, choice in enumerate(choices):
            if choice is None or site != choice.donor:
                continue
            if choices[choice.receiver] == choice:
                # Mutual choices imply vertex-disjoint matches within this round.
                agents[choice.donor].reserve(choice.quantity)
                agents[choice.receiver].reserve(choice.quantity)
                promise = Promise(round_index, choice.donor, choice.receiver,
                                  choice.quantity, choice.gain)
                promises.append(promise)
                confirmed.append(promise)
        trace.append({"round": round_index,
                      "messages": [message.__dict__ for message in messages],
                      "choices": [None if edge is None else edge.__dict__ for edge in choices],
                      "confirmed": [promise.__dict__ for promise in confirmed]})
        if not confirmed:
            break
    return promises, trace
