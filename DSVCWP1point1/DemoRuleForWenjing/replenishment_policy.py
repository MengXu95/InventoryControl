"""
Inventory Replenishment Policy Module

This module implements genetic programming-based replenishment policies for retailers.
It evaluates expression trees to determine optimal order quantities based on current
inventory state and various cost parameters.

Features:
- Safe mathematical operations with overflow/underflow protection
- Expression tree evaluation for complex replenishment policies
- Support for multiple state variables (inventory, costs, forecasts)
"""

import numpy as np


# State variable indices for clarity
STATE_INV_LEVEL = 0
STATE_HOLDING_COST = 1
STATE_LOST_SALES_COST = 2
STATE_CAPACITY = 3
STATE_FIXED_ORDER_COST = 4
STATE_PIPELINE = 5
STATE_FORECAST_1 = 6
STATE_FORECAST_2 = 7
STATE_TRANSSHIP_COST = 8
STATE_FIXED_TRANSSHIP_COST = 9


def parse_policy_string(policy_string):
    """
    Parse a policy string expression into a list format for tree evaluation.

    Converts nested function notation into a flat list representation suitable
    for tree traversal. Handles both quoted and unquoted terminal symbols.

    Args:
        policy_string: String representation of policy, e.g.,
            "subtract('FC1', 'INL')" or "add(subtract('FC1', 'INL'), 'PIP')"

    Returns:
        list: Flat list representation of the expression tree

    Examples:
        >> parse_policy_string("subtract('FC1', 'INL')")
        ['subtract', 'FC1', 'INL']

        >> parse_policy_string("add(subtract('FC1', 'INL'), 'PIP')")
        ['add', 'subtract', 'FC1', 'INL', 'PIP']
    """
    # Remove all whitespace for easier parsing
    policy_string = policy_string.replace(' ', '')

    result = []
    i = 0

    while i < len(policy_string):
        char = policy_string[i]

        # Skip commas and parentheses (structural elements)
        if char in ',()':
            i += 1
            continue

        # Handle quoted strings (terminal symbols)
        if char in '"\'':
            quote_char = char
            i += 1
            token = ''
            while i < len(policy_string) and policy_string[i] != quote_char:
                token += policy_string[i]
                i += 1
            i += 1  # Skip closing quote
            result.append(token)

        # Handle unquoted tokens (functions or terminals)
        else:
            token = ''
            while i < len(policy_string) and policy_string[i] not in ',()"\' ':
                token += policy_string[i]
                i += 1
            if token:  # Only add non-empty tokens
                result.append(token)

    return result


def calculate_replenishment_quantity(retailer, policy_expression=None):
    """
    Calculate the replenishment quantity for a retailer based on a policy expression.

    Args:
        retailer: Retailer object with attributes:
            - inv_level: Current inventory level
            - holding_cost: Cost per unit held in inventory
            - lost_sales_cost: Cost per unit of lost sales
            - capacity: Maximum inventory capacity
            - fixed_order_cost: Fixed cost per order
            - pipeline: List of orders in transit (assumes length >= 1)
            - forecast: Demand forecast list (assumes length >= 2)
            - transshipment_cost: Cost per unit transshipped
            - fixed_order_transshipment_cost: Fixed transshipment cost
        policy_expression: String representation of the policy tree.
            If None, uses a default policy.

    Returns:
        float: Recommended replenishment quantity, rounded to 2 decimal places

    Example:
        quantity = calculate_replenishment_quantity(my_retailer)
    """
    # Default policy if none provided
    if policy_expression is None:
        policy_expression = (
            "subtract(minimum(minimum(minimum('FC2', minimum(minimum('FC2', 'FC2'), "
            "minimum('FC2', 'FC2'))), 'FC2'), 'FC2'), minimum('PIP', 'INL'))"
        )

    # Convert string to list if necessary
    if isinstance(policy_expression, str):
        policy_expression = parse_policy_string(policy_expression)

    # Construct state vector
    state = np.array([
        retailer.inv_level,
        retailer.holding_cost,
        retailer.lost_sales_cost,
        retailer.capacity,
        retailer.fixed_order_cost,
        retailer.pipeline[0],  # First pipeline value (next delivery)
        retailer.forecast[0],   # First forecast period
        retailer.forecast[1],   # Second forecast period
        retailer.transshipment_cost,
        retailer.fixed_order_transshipment_cost
    ])

    replenishment_qty = _evaluate_policy_tree(state, policy_expression)
    return round(replenishment_qty, 0)


def _evaluate_policy_tree(state, tree_expression):
    """
    Evaluate a genetic programming tree expression.

    Args:
        state: numpy array of state variables
        tree_expression: String or parsed tree representation

    Returns:
        float: Evaluated result from the expression tree
    """
    inventory_replenishment, _ = _evaluate_tree_node(tree_expression, 0, state)
    return inventory_replenishment


def _evaluate_tree_node(tree, index, state):
    """
    Recursively evaluate a node in the expression tree.

    Args:
        tree: Expression tree (string or list)
        index: Current node index
        state: State vector for variable lookups

    Returns:
        tuple: (evaluated_value, length_of_subtree)
    """
    node = tree[index]

    # Binary operators
    if node == 'add':
        left, len_left = _evaluate_tree_node(tree, index + 1, state)
        right, len_right = _evaluate_tree_node(tree, index + len_left + 1, state)
        return _safe_add(left, right), len_left + len_right + 1

    elif node == 'subtract':
        left, len_left = _evaluate_tree_node(tree, index + 1, state)
        right, len_right = _evaluate_tree_node(tree, index + len_left + 1, state)
        return _safe_subtract(left, right), len_left + len_right + 1

    elif node == 'multiply':
        left, len_left = _evaluate_tree_node(tree, index + 1, state)
        right, len_right = _evaluate_tree_node(tree, index + len_left + 1, state)
        return _safe_multiply(left, right), len_left + len_right + 1

    elif node == 'protected_div':
        left, len_left = _evaluate_tree_node(tree, index + 1, state)
        right, len_right = _evaluate_tree_node(tree, index + len_left + 1, state)
        return _protected_div(left, right), len_left + len_right + 1

    elif node == 'maximum':
        left, len_left = _evaluate_tree_node(tree, index + 1, state)
        right, len_right = _evaluate_tree_node(tree, index + len_left + 1, state)
        return np.maximum(left, right), len_left + len_right + 1

    elif node == 'minimum':
        left, len_left = _evaluate_tree_node(tree, index + 1, state)
        right, len_right = _evaluate_tree_node(tree, index + len_left + 1, state)
        return np.minimum(left, right), len_left + len_right + 1

    # Unary operators
    elif node == 'protected_sqrt':
        child, len_child = _evaluate_tree_node(tree, index + 1, state)
        return _protected_sqrt(child), len_child + 1

    elif node == 'square':
        child, len_child = _evaluate_tree_node(tree, index + 1, state)
        return _safe_square(child), len_child + 1

    elif node == 'lf':  # Logistic function
        child, len_child = _evaluate_tree_node(tree, index + 1, state)
        return _logistic_function(child), len_child + 1

    # Terminal nodes (state variables)
    elif node == 'INL':  # Inventory Level
        return state[STATE_INV_LEVEL], 1
    elif node == 'PHC':  # Per-unit Holding Cost
        return state[STATE_HOLDING_COST], 1
    elif node == 'PLSC':  # Per-unit Lost Sales Cost
        return state[STATE_LOST_SALES_COST], 1
    elif node == 'INC':  # Inventory Capacity
        return state[STATE_CAPACITY], 1
    elif node == 'FOC':  # Fixed Order Cost
        return state[STATE_FIXED_ORDER_COST], 1
    elif node == 'PIP':  # Pipeline inventory
        return state[STATE_PIPELINE], 1
    elif node == 'FC1':  # Forecast period 1
        return state[STATE_FORECAST_1], 1
    elif node == 'FC2':  # Forecast period 2
        return state[STATE_FORECAST_2], 1
    elif node == 'PTC':  # Per-unit Transshipment Cost
        return state[STATE_TRANSSHIP_COST], 1
    elif node == 'FTC':  # Fixed Transshipment Cost
        return state[STATE_FIXED_TRANSSHIP_COST], 1

    else:
        raise ValueError(f"Unknown node type: {node}")


# Safe mathematical operations with overflow/underflow protection

def _protected_div(left, right):
    """Division with protection against division by zero and invalid operations."""
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x = np.where(np.isinf(x) | np.isnan(x), 1, x)
        else:
            x = 1 if np.isinf(x) or np.isnan(x) else x
    return x


def _protected_sqrt(x):
    """Square root with protection for negative values."""
    if x > 0:
        value = np.sqrt(x)
        return np.inf if np.isnan(value) else value
    return 0.0


def _safe_multiply(val1, val2):
    """Multiplication with overflow protection."""
    try:
        val1 = float(val1)
        val2 = float(val2)

        if np.isinf(val1) or np.isnan(val1) or np.isinf(val2) or np.isnan(val2):
            return np.inf

        result = val1 * val2
        return np.inf if np.isinf(result) or np.isnan(result) else result
    except (ValueError, TypeError):
        return np.inf


def _safe_subtract(a, b):
    """Subtraction with overflow protection."""
    with np.errstate(over='ignore', invalid='ignore'):
        value = np.subtract(a, b)
        if np.isinf(value) or np.isnan(value):
            value = np.inf
    return value


def _safe_add(a, b):
    """Addition with overflow protection."""
    with np.errstate(over='ignore', invalid='ignore'):
        value = np.add(a, b)
        if np.isinf(value) or np.isnan(value):
            value = np.inf
    return value


def _safe_square(a):
    """Squaring operation with overflow protection."""
    try:
        a = float(a)
        if np.isinf(a) or np.isnan(a):
            return np.inf

        result = np.square(a)
        return np.inf if np.isinf(result) or np.isnan(result) else result
    except (ValueError, TypeError):
        return np.inf


def _logistic_function(x):
    """
    Apply logistic (sigmoid) function: 1 / (1 + exp(-x))

    Handles both scalar and array inputs.
    """
    if isinstance(x, (np.int64, np.float64, float, int)):
        return 1 / (1 + np.exp(-x))
    else:
        # Apply element-wise for arrays
        return 1 / (1 + np.exp(-np.array(x)))


# Example Retailer class for demonstration
class Retailer:
    """
    Simple retailer class for demonstration purposes.

    In production, this would be replaced with your actual retailer implementation.
    """
    def __init__(self, inv_level, holding_cost, lost_sales_cost, capacity,
                 fixed_order_cost, pipeline, forecast, transshipment_cost,
                 fixed_order_transshipment_cost):
        self.inv_level = inv_level
        self.holding_cost = holding_cost
        self.lost_sales_cost = lost_sales_cost
        self.capacity = capacity
        self.fixed_order_cost = fixed_order_cost
        self.pipeline = pipeline
        self.forecast = forecast
        self.transshipment_cost = transshipment_cost
        self.fixed_order_transshipment_cost = fixed_order_transshipment_cost


def main():
    """
    Main function demonstrating the replenishment policy calculation.

    This example creates a sample retailer and calculates the recommended
    replenishment quantity using both default and custom policies.
    """
    print("=" * 70)
    print("Inventory Replenishment Policy Demonstration")
    print("=" * 70)

    # Create a sample retailer with typical parameters
    sample_retailer = Retailer(
        inv_level=50.0,              # Current inventory: 50 units
        holding_cost=2.5,            # $2.50 per unit held
        lost_sales_cost=15.0,        # $15 per unit of lost sales
        capacity=200.0,              # Maximum capacity: 200 units
        fixed_order_cost=100.0,      # $100 fixed cost per order
        pipeline=[30.0, 20.0],       # 30 units arriving next, 20 after
        forecast=[80.0, 75.0],       # Forecast: 80 units next period, 75 after
        transshipment_cost=5.0,      # $5 per unit transshipped
        fixed_order_transshipment_cost=50.0  # $50 fixed transshipment cost
    )

    print("\nRetailer State:")
    print(f"  Current Inventory Level: {sample_retailer.inv_level} units")
    print(f"  Holding Cost: ${sample_retailer.holding_cost}/unit")
    print(f"  Lost Sales Cost: ${sample_retailer.lost_sales_cost}/unit")
    print(f"  Capacity: {sample_retailer.capacity} units")
    print(f"  Fixed Order Cost: ${sample_retailer.fixed_order_cost}")
    print(f"  Pipeline Inventory: {sample_retailer.pipeline}")
    print(f"  Demand Forecast: {sample_retailer.forecast}")
    print(f"  Transshipment Cost: ${sample_retailer.transshipment_cost}/unit")
    print(f"  Fixed Transshipment Cost: ${sample_retailer.fixed_order_transshipment_cost}")

    # Example 1: Using default policy
    print("\n" + "-" * 70)
    print("Example 1: Default Policy")
    print("-" * 70)

    replenishment_qty = calculate_replenishment_quantity(sample_retailer)
    print(f"\nRecommended Replenishment Quantity: {replenishment_qty} units")

    # Example 2: Using a custom simple policy
    print("\n" + "-" * 70)
    print("Example 2: Custom Policy - Order based on forecast minus inventory")
    print("-" * 70)

    custom_policy = "subtract('FC1', 'INL')"
    replenishment_qty_custom = calculate_replenishment_quantity(
        sample_retailer,
        policy_expression=custom_policy
    )
    print(f"Policy Expression: {custom_policy}")
    print(f"Recommended Replenishment Quantity: {replenishment_qty_custom} units")
    print(f"  (Forecast period 1: {sample_retailer.forecast[0]} - " +
          f"Current inventory: {sample_retailer.inv_level} = {replenishment_qty_custom})")

    # Example 3: Using a maximum policy
    print("\n" + "-" * 70)
    print("Example 3: Maximum of two forecasts")
    print("-" * 70)

    max_policy = "maximum('FC1', 'FC2')"
    replenishment_qty_max = calculate_replenishment_quantity(
        sample_retailer,
        policy_expression=max_policy
    )
    print(f"Policy Expression: {max_policy}")
    print(f"Recommended Replenishment Quantity: {replenishment_qty_max} units")
    print(f"  (Max of FC1: {sample_retailer.forecast[0]} and " +
          f"FC2: {sample_retailer.forecast[1]} = {replenishment_qty_max})")

    # Example 4: Complex policy with multiple operations
    print("\n" + "-" * 70)
    print("Example 4: Complex Policy")
    print("-" * 70)

    complex_policy = "subtract(add('FC1', 'FC2'), add('INL', 'PIP'))"
    replenishment_qty_complex = calculate_replenishment_quantity(
        sample_retailer,
        policy_expression=complex_policy
    )
    print(f"Policy Expression: {complex_policy}")
    print(f"  (Total forecast - Current inventory - Pipeline)")
    print(f"Recommended Replenishment Quantity: {replenishment_qty_complex} units")

    print("\n" + "=" * 70)
    print("Available State Variables:")
    print("=" * 70)
    print("  INL  - Inventory Level")
    print("  PHC  - Per-unit Holding Cost")
    print("  PLSC - Per-unit Lost Sales Cost")
    print("  INC  - Inventory Capacity")
    print("  FOC  - Fixed Order Cost")
    print("  PIP  - Pipeline inventory")
    print("  FC1  - Forecast period 1")
    print("  FC2  - Forecast period 2")
    print("  PTC  - Per-unit Transshipment Cost")
    print("  FTC  - Fixed Transshipment Cost")

    print("\nAvailable Operations:")
    print("  Binary: add, subtract, multiply, protected_div, maximum, minimum")
    print("  Unary: protected_sqrt, square, lf (logistic function)")
    print("=" * 70)


if __name__ == "__main__":
    main()


