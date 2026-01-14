"""
Inventory Transshipment Policy Module

This module implements genetic programming-based transshipment policies for
retailer pairs. It evaluates expression trees to determine optimal transshipment
quantities between two retailers based on their inventory states and cost parameters.

Features:
- Safe mathematical operations with overflow/underflow protection
- Expression tree evaluation for complex transshipment policies
- Support for two-retailer state variables (inventory, costs, forecasts)
- Policy string parsing for user-friendly policy definition
"""

import numpy as np

# State variable indices for retailer pair
STATE_RETAILER_I_ID = 0
STATE_RETAILER_J_ID = 1
STATE_INV_LEVEL_1 = 2
STATE_HOLDING_COST_1 = 3
STATE_LOST_SALES_COST_1 = 4
STATE_CAPACITY_1 = 5
STATE_FIXED_ORDER_COST_1 = 6
STATE_PIPELINE_1 = 7
STATE_FORECAST_1_1 = 8
STATE_FORECAST_1_2 = 9
STATE_INV_LEVEL_2 = 10
STATE_HOLDING_COST_2 = 11
STATE_LOST_SALES_COST_2 = 12
STATE_CAPACITY_2 = 13
STATE_FIXED_ORDER_COST_2 = 14
STATE_PIPELINE_2 = 15
STATE_FORECAST_2_1 = 16
STATE_FORECAST_2_2 = 17
STATE_TRANSSHIP_COST = 18
STATE_FIXED_TRANSSHIP_COST = 19


def parse_policy_string(policy_string):
    """
    Parse a policy string expression into a list format for tree evaluation.

    Converts nested function notation into a flat list representation suitable
    for tree traversal. Handles both quoted and unquoted terminal symbols.

    Args:
        policy_string: String representation of policy, e.g.,
            "subtract('INL1', 'INL2')" or "add(subtract('FC11', 'INL1'), 'PIP1')"

    Returns:
        list: Flat list representation of the expression tree

    Examples:
        >> parse_policy_string("subtract('INL1', 'INL2')")
        ['subtract', 'INL1', 'INL2']

        >> parse_policy_string("maximum('INL1', 'INL2')")
        ['maximum', 'INL1', 'INL2']
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


def calculate_transshipment_quantity(retailer_i, retailer_j,
                                     retailer_i_id=0, retailer_j_id=1,
                                     policy_expression=None):
    """
    Calculate the transshipment quantity between two retailers based on a policy expression.

    Args:
        retailer_i: First retailer object with attributes:
            - inv_level: Current inventory level
            - holding_cost: Cost per unit held in inventory
            - lost_sales_cost: Cost per unit of lost sales
            - capacity: Maximum inventory capacity
            - fixed_order_cost: Fixed cost per order
            - pipeline: List of orders in transit (assumes length >= 1)
            - forecast: Demand forecast list (assumes length >= 2)
        retailer_j: Second retailer object with attributes:
            - inv_level, holding_cost, lost_sales_cost, capacity, fixed_order_cost
            - pipeline, forecast (same as retailer_i)
            - transshipment_cost: Cost per unit transshipped
            - fixed_order_transshipment_cost: Fixed transshipment cost
        retailer_i_id: Identifier for retailer i (default: 0)
        retailer_j_id: Identifier for retailer j (default: 1)
        policy_expression: String or list representation of the policy tree.
            If None, uses a default policy. Can be either:
            - String: "subtract('INL1', 'INL2')"
            - List: ['subtract', 'INL1', 'INL2']

    Returns:
        float: Recommended transshipment quantity from retailer i to retailer j,
               rounded to 2 decimal places. Positive values indicate shipment
               from i to j, negative values indicate shipment from j to i.

    Example:
        >> quantity = calculate_transshipment_quantity(retailer_a, retailer_b)
        >> quantity = calculate_transshipment_quantity(
        ...     retailer_a, retailer_b,
        ...     policy_expression="subtract('INL1', 'INL2')"
        ... )
    """
    # Default policy if none provided
    if policy_expression is None:
        policy_expression = (
            "square(multiply('PLSC2', multiply('PHC1', "
            "protected_div('FOC1', multiply('FTC', 'FTC')))))"
        )

    # Convert string to list if necessary
    if isinstance(policy_expression, str):
        policy_expression = parse_policy_string(policy_expression)

    # Construct state vector for retailer pair
    state = np.array([
        retailer_i_id,  # Retailer i ID
        retailer_j_id,  # Retailer j ID
        retailer_i.inv_level,  # Retailer 1 inventory level
        retailer_i.holding_cost,  # Retailer 1 holding cost
        retailer_i.lost_sales_cost,  # Retailer 1 lost sales cost
        retailer_i.capacity,  # Retailer 1 capacity
        retailer_i.fixed_order_cost,  # Retailer 1 fixed order cost
        retailer_i.pipeline[0],  # Retailer 1 pipeline (next delivery)
        retailer_i.forecast[0],  # Retailer 1 forecast period 1
        retailer_i.forecast[1],  # Retailer 1 forecast period 2
        retailer_j.inv_level,  # Retailer 2 inventory level
        retailer_j.holding_cost,  # Retailer 2 holding cost
        retailer_j.lost_sales_cost,  # Retailer 2 lost sales cost
        retailer_j.capacity,  # Retailer 2 capacity
        retailer_j.fixed_order_cost,  # Retailer 2 fixed order cost
        retailer_j.pipeline[0],  # Retailer 2 pipeline (next delivery)
        retailer_j.forecast[0],  # Retailer 2 forecast period 1
        retailer_j.forecast[1],  # Retailer 2 forecast period 2
        retailer_j.transshipment_cost,  # Per-unit transshipment cost
        retailer_j.fixed_order_transshipment_cost  # Fixed transshipment cost
    ])

    transshipment_qty = _evaluate_policy_tree(state, policy_expression)
    return round(transshipment_qty, 0)


def _evaluate_policy_tree(state, tree_expression):
    """
    Evaluate a genetic programming tree expression.

    Args:
        state: numpy array of state variables for retailer pair
        tree_expression: String or parsed tree representation

    Returns:
        float: Evaluated result from the expression tree
    """
    transshipment_quantity, _ = _evaluate_tree_node(tree_expression, 0, state)
    return transshipment_quantity


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

    # Terminal nodes - Retailer 1 (i) variables
    elif node == 'INL1':  # Inventory Level - Retailer 1
        return state[STATE_INV_LEVEL_1], 1
    elif node == 'PHC1':  # Per-unit Holding Cost - Retailer 1
        return state[STATE_HOLDING_COST_1], 1
    elif node == 'PLSC1':  # Per-unit Lost Sales Cost - Retailer 1
        return state[STATE_LOST_SALES_COST_1], 1
    elif node == 'INC1':  # Inventory Capacity - Retailer 1
        return state[STATE_CAPACITY_1], 1
    elif node == 'FOC1':  # Fixed Order Cost - Retailer 1
        return state[STATE_FIXED_ORDER_COST_1], 1
    elif node == 'PIP1':  # Pipeline inventory - Retailer 1
        return state[STATE_PIPELINE_1], 1
    elif node == 'FC11':  # Forecast period 1 - Retailer 1
        return state[STATE_FORECAST_1_1], 1
    elif node == 'FC12':  # Forecast period 2 - Retailer 1
        return state[STATE_FORECAST_1_2], 1

    # Terminal nodes - Retailer 2 (j) variables
    elif node == 'INL2':  # Inventory Level - Retailer 2
        return state[STATE_INV_LEVEL_2], 1
    elif node == 'PHC2':  # Per-unit Holding Cost - Retailer 2
        return state[STATE_HOLDING_COST_2], 1
    elif node == 'PLSC2':  # Per-unit Lost Sales Cost - Retailer 2
        return state[STATE_LOST_SALES_COST_2], 1
    elif node == 'INC2':  # Inventory Capacity - Retailer 2
        return state[STATE_CAPACITY_2], 1
    elif node == 'FOC2':  # Fixed Order Cost - Retailer 2
        return state[STATE_FIXED_ORDER_COST_2], 1
    elif node == 'PIP2':  # Pipeline inventory - Retailer 2
        return state[STATE_PIPELINE_2], 1
    elif node == 'FC21':  # Forecast period 1 - Retailer 2
        return state[STATE_FORECAST_2_1], 1
    elif node == 'FC22':  # Forecast period 2 - Retailer 2
        return state[STATE_FORECAST_2_2], 1

    # Shared transshipment variables
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
                 fixed_order_cost, pipeline, forecast, transshipment_cost=0,
                 fixed_order_transshipment_cost=0):
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
    Main function demonstrating the transshipment policy calculation.

    This example creates two sample retailers and calculates the recommended
    transshipment quantity using both default and custom policies.
    """
    print("=" * 70)
    print("Inventory Transshipment Policy Demonstration")
    print("=" * 70)

    # Create two sample retailers with different inventory situations
    retailer_a = Retailer(
        inv_level=80.0,  # High inventory: 80 units
        holding_cost=2.0,  # $2 per unit held
        lost_sales_cost=12.0,  # $12 per unit of lost sales
        capacity=200.0,  # Maximum capacity: 200 units
        fixed_order_cost=100.0,  # $100 fixed cost per order
        pipeline=[20.0, 15.0],  # 20 units arriving next, 15 after
        forecast=[60.0, 55.0],  # Lower forecast: 60 and 55 units
        transshipment_cost=3.0,  # $3 per unit transshipped
        fixed_order_transshipment_cost=40.0  # $40 fixed transshipment cost
    )

    retailer_b = Retailer(
        inv_level=25.0,  # Low inventory: 25 units
        holding_cost=2.5,  # $2.50 per unit held
        lost_sales_cost=18.0,  # $18 per unit of lost sales
        capacity=180.0,  # Maximum capacity: 180 units
        fixed_order_cost=100.0,  # $100 fixed cost per order
        pipeline=[10.0, 12.0],  # 10 units arriving next, 12 after
        forecast=[90.0, 85.0],  # Higher forecast: 90 and 85 units
        transshipment_cost=3.0,  # $3 per unit transshipped
        fixed_order_transshipment_cost=40.0  # $40 fixed transshipment cost
    )

    print("\nRetailer A State:")
    print(f"  Current Inventory Level: {retailer_a.inv_level} units")
    print(f"  Holding Cost: ${retailer_a.holding_cost}/unit")
    print(f"  Lost Sales Cost: ${retailer_a.lost_sales_cost}/unit")
    print(f"  Pipeline Inventory: {retailer_a.pipeline}")
    print(f"  Demand Forecast: {retailer_a.forecast}")

    print("\nRetailer B State:")
    print(f"  Current Inventory Level: {retailer_b.inv_level} units")
    print(f"  Holding Cost: ${retailer_b.holding_cost}/unit")
    print(f"  Lost Sales Cost: ${retailer_b.lost_sales_cost}/unit")
    print(f"  Pipeline Inventory: {retailer_b.pipeline}")
    print(f"  Demand Forecast: {retailer_b.forecast}")
    print(f"  Transshipment Cost: ${retailer_b.transshipment_cost}/unit")
    print(f"  Fixed Transshipment Cost: ${retailer_b.fixed_order_transshipment_cost}")

    # Demonstrate policy string parsing
    print("\n" + "-" * 70)
    print("Policy String Parsing Demo")
    print("-" * 70)

    test_policies = [
        "subtract('INL1', 'INL2')",
        "multiply(protected_div('INL1', 'INL2'), 'PTC')",
        "maximum('FC11', 'FC21')"
    ]

    for policy_str in test_policies:
        parsed = parse_policy_string(policy_str)
        print(f"\nOriginal: {policy_str}")
        print(f"Parsed:   {parsed}")

    # Example 1: Using default policy
    print("\n" + "-" * 70)
    print("Example 1: Default Policy")
    print("-" * 70)

    transship_qty = calculate_transshipment_quantity(retailer_a, retailer_b)
    print(f"\nRecommended Transshipment Quantity: {transship_qty} units")
    if transship_qty > 0:
        print(f"  → Ship {transship_qty} units from Retailer A to Retailer B")
    elif transship_qty < 0:
        print(f"  → Ship {abs(transship_qty)} units from Retailer B to Retailer A")
    else:
        print("  → No transshipment recommended")

    # Example 2: Simple policy - ship based on inventory difference
    print("\n" + "-" * 70)
    print("Example 2: Inventory Difference Policy")
    print("-" * 70)

    simple_policy = "multiply(subtract('INL1', 'INL2'), 0.5)"
    # Note: This would need numeric literal support, so using a simpler version
    simple_policy = "subtract('INL1', 'INL2')"
    transship_qty_simple = calculate_transshipment_quantity(
        retailer_a, retailer_b,
        policy_expression=simple_policy
    )
    print(f"Policy Expression: {simple_policy}")
    print(f"Recommended Transshipment Quantity: {transship_qty_simple} units")
    print(f"  (Retailer A inventory: {retailer_a.inv_level} - " +
          f"Retailer B inventory: {retailer_b.inv_level} = {transship_qty_simple})")

    # Example 3: Maximum forecast-based policy
    print("\n" + "-" * 70)
    print("Example 3: Forecast-Based Policy")
    print("-" * 70)

    forecast_policy = "subtract('FC21', 'FC11')"
    transship_qty_forecast = calculate_transshipment_quantity(
        retailer_a, retailer_b,
        policy_expression=forecast_policy
    )
    print(f"Policy Expression: {forecast_policy}")
    print(f"Recommended Transshipment Quantity: {transship_qty_forecast} units")
    print(f"  (Retailer B forecast: {retailer_b.forecast[0]} - " +
          f"Retailer A forecast: {retailer_a.forecast[0]} = {transship_qty_forecast})")

    # Example 4: Complex policy considering multiple factors
    print("\n" + "-" * 70)
    print("Example 4: Complex Multi-Factor Policy")
    print("-" * 70)

    complex_policy = "subtract(subtract('INL1', 'FC11'), subtract('INL2', 'FC21'))"
    transship_qty_complex = calculate_transshipment_quantity(
        retailer_a, retailer_b,
        policy_expression=complex_policy
    )
    print(f"Policy Expression: {complex_policy}")
    print(f"  ((INL1 - FC11) - (INL2 - FC21))")
    print(f"  (Excess/deficit at A) - (Excess/deficit at B)")
    print(f"Recommended Transshipment Quantity: {transship_qty_complex} units")

    print("\n" + "=" * 70)
    print("Available State Variables:")
    print("=" * 70)
    print("\nRetailer 1 (Source) Variables:")
    print("  INL1  - Inventory Level")
    print("  PHC1  - Per-unit Holding Cost")
    print("  PLSC1 - Per-unit Lost Sales Cost")
    print("  INC1  - Inventory Capacity")
    print("  FOC1  - Fixed Order Cost")
    print("  PIP1  - Pipeline inventory")
    print("  FC11  - Forecast period 1")
    print("  FC12  - Forecast period 2")

    print("\nRetailer 2 (Destination) Variables:")
    print("  INL2  - Inventory Level")
    print("  PHC2  - Per-unit Holding Cost")
    print("  PLSC2 - Per-unit Lost Sales Cost")
    print("  INC2  - Inventory Capacity")
    print("  FOC2  - Fixed Order Cost")
    print("  PIP2  - Pipeline inventory")
    print("  FC21  - Forecast period 1")
    print("  FC22  - Forecast period 2")

    print("\nShared Variables:")
    print("  PTC   - Per-unit Transshipment Cost")
    print("  FTC   - Fixed Transshipment Cost")

    print("\nAvailable Operations:")
    print("  Binary: add, subtract, multiply, protected_div, maximum, minimum")
    print("  Unary: protected_sqrt, square, lf (logistic function)")

    print("\nPolicy Format:")
    print("  String: \"subtract('INL1', 'INL2')\"")
    print("  List:   ['subtract', 'INL1', 'INL2']")

    print("\nInterpretation:")
    print("  Positive values: Ship from Retailer 1 (i) to Retailer 2 (j)")
    print("  Negative values: Ship from Retailer 2 (j) to Retailer 1 (i)")
    print("  Zero: No transshipment recommended")
    print("=" * 70)


if __name__ == "__main__":
    main()




