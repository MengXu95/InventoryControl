"""Role-specific DEAP trees; bounded arithmetic and canonical, validated JSON."""

import ast
import copy
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

from deap import gp

ROLES = ("order", "request", "offer")
COMMON = ("I", "PIPE", "DUE", "F", "FC1", "FC2", "CAP", "H", "B",
          "RHO", "LAST_IN", "LAST_OUT", "LAST_LOST")
TERMINALS = {
    "order": COMMON + ("ORDER_FIXED", "ORDER_UNIT"),
    "request": COMMON + ("ROOM", "DEFICIT", "TC", "TK"),
    "offer": COMMON + ("AVAIL", "SURPLUS", "TC", "TK"),
}
BOUND = 1_000_000.0
MAX_HEIGHT = 5
MAX_NODES = 63  # per tree
SCHEMA = "three-tree-peer-v1"


class NumericalPolicyError(ValueError):
    """Invalid nonfinite arithmetic is an experiment failure, not a zero action."""


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise NumericalPolicyError(f"Nonfinite policy value: {value}")
    return value


def bounded(value):
    return max(-BOUND, min(BOUND, finite(value)))


def add(a, b):
    return bounded(a + b)


def sub(a, b):
    return bounded(a - b)


def mul(a, b):
    return bounded(a * b)


def div(a, b):
    return 1.0 if abs(b) <= 1e-8 else bounded(a / b)


def maximum(a, b):
    return bounded(max(a, b))


def minimum(a, b):
    return bounded(min(a, b))


PRIMITIVES = {"add": add, "sub": sub, "mul": mul, "div": div,
              "maximum": maximum, "minimum": minimum}


def primitive_set(role):
    names = TERMINALS[role]
    pset = gp.PrimitiveSet(role, len(names))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(names)})
    for name, function in PRIMITIVES.items():
        pset.addPrimitive(function, 2, name=name)
    for constant in (0.0, 0.25, 0.5, 1.0, 2.0):
        pset.addTerminal(constant)
    return pset


PSETS = {role: primitive_set(role) for role in ROLES}


def parse_tree(expression, role):
    """Reject arbitrary Python before DEAP's parser/compiler ever sees a string."""
    if not isinstance(expression, str) or len(expression) > 8000:
        raise ValueError("Tree must be a short expression string")
    try:
        root = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid tree syntax") from exc

    def check(node, depth=0):
        if depth > MAX_HEIGHT:
            raise ValueError("Tree height limit exceeded")
        if isinstance(node, ast.Call):
            if (not isinstance(node.func, ast.Name)
                    or node.func.id not in PRIMITIVES
                    or len(node.args) != 2 or node.keywords):
                raise ValueError("Unknown primitive or arity")
            for child in node.args:
                check(child, depth + 1)
        elif isinstance(node, ast.Name):
            if node.id not in TERMINALS[role]:
                raise ValueError(f"Unknown {role} terminal: {node.id}")
        elif isinstance(node, ast.Constant) and type(node.value) in (float, int):
            if not math.isfinite(node.value) or not 0 <= node.value <= BOUND:
                raise ValueError("Constants must be finite and in [0, 1e6]")
        else:
            raise ValueError("Only primitives, local terminals and numeric constants allowed")

    check(root.body)
    tree = gp.PrimitiveTree.from_string(expression, PSETS[role])
    if len(tree) > MAX_NODES or tree.height > MAX_HEIGHT:
        raise ValueError("Tree size/height limit exceeded")
    return tree


class Individual(list):
    """Exactly three trees, in fixed order/request/offer module order."""

    def __init__(self, trees):
        if len(trees) != 3:
            raise ValueError("An individual must have exactly three trees")
        super().__init__([parse_tree(str(tree), role) for role, tree in zip(ROLES, trees)])

    @property
    def size(self):
        return sum(map(len, self))

    def expressions(self):
        return dict(zip(ROLES, map(str, self)))


def template():
    return Individual(["maximum(0.0, sub(F, add(I, PIPE)))", "DEFICIT", "SURPLUS"])


def random_individual():
    return Individual([gp.PrimitiveTree(gp.genHalfAndHalf(
        pset=PSETS[role], min_=0, max_=2)) for role in ROLES])


def crossover(left, right):
    """One homologous module only; revert a child violating bloat limits."""
    children = [copy.deepcopy(left), copy.deepcopy(right)]
    module = random.randrange(3)
    gp.cxOnePoint(children[0][module], children[1][module])
    for index, parent in enumerate((left, right)):
        tree = children[index][module]
        if tree.height > MAX_HEIGHT or len(tree) > MAX_NODES:
            children[index] = copy.deepcopy(parent)
    return tuple(Individual(child) for child in children)


def mutate(parent):
    child = copy.deepcopy(parent)
    module = random.randrange(3)
    role = ROLES[module]
    gp.mutUniform(child[module],
                  expr=lambda pset, type_: gp.genGrow(pset, 0, 2, type_),
                  pset=PSETS[role])
    if child[module].height > MAX_HEIGHT or len(child[module]) > MAX_NODES:
        return copy.deepcopy(parent)
    return Individual(child)


@contextmanager
def seeded_random(seed):
    """DEAP uses stdlib random; restore caller state on exit."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class Policy:
    def __init__(self, individual):
        self.individual = Individual(individual)
        self.functions = {role: gp.compile(tree, PSETS[role])
                          for role, tree in zip(ROLES, self.individual)}

    def output(self, role, features):
        # All inputs are local observation/memory, never a site index or mailbox.
        values = [bounded(features[name]) for name in TERMINALS[role]]
        try:
            return finite(self.functions[role](*values))
        except (ArithmeticError, TypeError) as exc:
            raise NumericalPolicyError(f"{role} tree failed") from exc


def policy_document(individual):
    return {"schema": SCHEMA, "trees": individual.expressions(),
            "terminals": {role: list(names) for role, names in TERMINALS.items()},
            "arithmetic_bound": BOUND, "max_height": MAX_HEIGHT,
            "max_nodes_per_tree": MAX_NODES}


def canonical_json(document):
    return json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"


def save_policy(individual, path):
    Path(path).write_text(canonical_json(policy_document(individual)), encoding="utf-8")


def load_policy(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or set(data.get("trees", {})) != set(ROLES):
        raise ValueError("Expected canonical three-tree-peer-v1 policy")
    individual = Individual([data["trees"][role] for role in ROLES])
    if data != policy_document(individual):
        raise ValueError("Policy metadata/terminal/arithmetic contract mismatch")
    return individual
