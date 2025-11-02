"""
Price Prediction Policy Module

This module implements genetic programming-based price prediction policies.
It evaluates expression trees to predict prices based on RFQ (Request for Quote)
and TUD (Time Until Delivery) variables.

Features:
- Safe mathematical operations with overflow/underflow protection
- Expression tree evaluation for complex price prediction policies
- Support for RFQ and TUD state variables
"""

import numpy as np


# State variable indices for clarity
STATE_RFQ = 0  # Request for Quote
STATE_TUD = 1  # Time Until Delivery


def parse_policy_string(policy_string):
    """
    Parse a policy string expression into a list format for tree evaluation.

    Converts nested function notation into a flat list representation suitable
    for tree traversal. Handles both quoted and unquoted terminal symbols.

    Args:
        policy_string: String representation of policy, e.g.,
            "protected_div('RFQ', 'TUD')" or "add('RFQ', 'TUD')"

    Returns:
        list: Flat list representation of the expression tree

    Examples:
        >> parse_policy_string("protected_div('RFQ', 'TUD')")
        ['protected_div', 'RFQ', 'TUD']

        >> parse_policy_string("add(multiply('RFQ', 'TUD'), 'RFQ')")
        ['add', 'multiply', 'RFQ', 'TUD', 'RFQ']
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


def predict_price(rfq, tud, policy_expression=None):
    """
    Predict price based on RFQ and TUD using a policy expression.

    Args:
        rfq: Request for Quote value (float)
        tud: Time Until Delivery value (float)
        policy_expression: String representation of the policy tree.
            If None, uses a default policy (protected_div('RFQ', 'TUD')).

    Returns:
        float: Predicted price, rounded to 2 decimal places

    Example:
        price = predict_price(rfq=100.0, tud=5.0)
    """
    # Default policy if none provided
    if policy_expression is None:
        policy_expression = "protected_div('RFQ', 'TUD')"

    # Convert string to list if necessary
    if isinstance(policy_expression, str):
        policy_expression = parse_policy_string(policy_expression)

    # Construct state vector (data)
    state = np.array([rfq, tud])

    predicted_price = _evaluate_policy_tree(state, policy_expression)
    return round(predicted_price, 2)


def _evaluate_policy_tree(state, tree_expression):
    """
    Evaluate a genetic programming tree expression for price prediction.

    Args:
        state: numpy array of state variables [RFQ, TUD]
        tree_expression: String or parsed tree representation

    Returns:
        float: Evaluated result from the expression tree
    """
    price_prediction, _ = _evaluate_tree_node(tree_expression, 0, state)
    return price_prediction


def _evaluate_tree_node(tree, index, state):
    """
    Recursively evaluate a node in the expression tree.

    This function corresponds to treeNode_RFQ_predict from the original code.

    Args:
        tree: Expression tree (string or list)
        index: Current node index
        state: State vector for variable lookups [RFQ, TUD]

    Returns:
        tuple: (evaluated_value, length_of_subtree)
    """
    node = tree[index]

    # Binary operators (arity == 2)
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

    # Unary operators (arity == 1)
    elif node == 'protected_sqrt':
        child, len_child = _evaluate_tree_node(tree, index + 1, state)
        return _protected_sqrt(child), len_child + 1

    elif node == 'square':
        child, len_child = _evaluate_tree_node(tree, index + 1, state)
        return _safe_square(child), len_child + 1

    elif node == 'lf':  # Logistic function
        child, len_child = _evaluate_tree_node(tree, index + 1, state)
        return _logistic_function(child), len_child + 1

    # Terminal nodes (state variables) - arity == 0
    elif node == 'RFQ':  # Request for Quote
        return state[STATE_RFQ], 1
    elif node == 'TUD':  # Time Until Delivery
        return state[STATE_TUD], 1

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


def main():
    """
    Main function demonstrating the price prediction policy calculation.

    This example shows how to use different policy expressions to predict
    prices based on RFQ (Request for Quote) and TUD (Time Until Delivery).
    """
    print("=" * 70)
    print("Price Prediction Policy Demonstration")
    print("=" * 70)

    # Sample input values
    sample_rfq = 100.0
    sample_tud = 5.0

    print("\nInput Variables:")
    print(f"  RFQ (Request for Quote): {sample_rfq}")
    print(f"  TUD (Time Until Delivery): {sample_tud}")

    # Example 1: Using default policy (protected_div)
    print("\n" + "-" * 70)
    print("Example 1: Default Policy - RFQ divided by TUD")
    print("-" * 70)

    predicted_price = predict_price(sample_rfq, sample_tud)
    print(f"Policy Expression: protected_div('RFQ', 'TUD')")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"  ({sample_rfq} / {sample_tud} = {predicted_price:.2f})")

    # Example 2: Simple addition policy
    print("\n" + "-" * 70)
    print("Example 2: Addition Policy - RFQ plus TUD")
    print("-" * 70)

    policy_add = "add('RFQ', 'TUD')"
    predicted_price_add = predict_price(sample_rfq, sample_tud, policy_expression=policy_add)
    print(f"Policy Expression: {policy_add}")
    print(f"Predicted Price: ${predicted_price_add:.2f}")
    print(f"  ({sample_rfq} + {sample_tud} = {predicted_price_add:.2f})")

    # Example 3: Multiplication policy
    print("\n" + "-" * 70)
    print("Example 3: Multiplication Policy - RFQ times TUD")
    print("-" * 70)

    policy_mult = "multiply('RFQ', 'TUD')"
    predicted_price_mult = predict_price(sample_rfq, sample_tud, policy_expression=policy_mult)
    print(f"Policy Expression: {policy_mult}")
    print(f"Predicted Price: ${predicted_price_mult:.2f}")
    print(f"  ({sample_rfq} * {sample_tud} = {predicted_price_mult:.2f})")

    # Example 4: Complex policy
    print("\n" + "-" * 70)
    print("Example 4: Complex Policy - (RFQ * TUD) / (RFQ + TUD)")
    print("-" * 70)

    complex_policy = "protected_div(multiply('RFQ', 'TUD'), add('RFQ', 'TUD'))"
    predicted_price_complex = predict_price(
        sample_rfq,
        sample_tud,
        policy_expression=complex_policy
    )
    print(f"Policy Expression: {complex_policy}")
    print(f"Predicted Price: ${predicted_price_complex:.2f}")
    numerator = sample_rfq * sample_tud
    denominator = sample_rfq + sample_tud
    print(f"  (({sample_rfq} * {sample_tud}) / ({sample_rfq} + {sample_tud}) = " +
          f"{numerator:.2f} / {denominator:.2f} = {predicted_price_complex:.2f})")

    # Example 5: Using logistic function
    print("\n" + "-" * 70)
    print("Example 5: Logistic Function Policy")
    print("-" * 70)

    logistic_policy = "multiply(lf(protected_div('RFQ', 'TUD')), 'RFQ')"
    predicted_price_logistic = predict_price(
        sample_rfq,
        sample_tud,
        policy_expression=logistic_policy
    )
    print(f"Policy Expression: {logistic_policy}")
    print(f"  Apply logistic function to (RFQ/TUD), then multiply by RFQ")
    print(f"Predicted Price: ${predicted_price_logistic:.2f}")

    # Example 6: Different input values
    print("\n" + "-" * 70)
    print("Example 6: Testing with Different Input Values")
    print("-" * 70)

    test_cases = [
        (150.0, 10.0),
        (80.0, 4.0),
        (200.0, 8.0),
        (50.0, 2.5)
    ]

    default_policy = "protected_div('RFQ', 'TUD')"
    print(f"Using policy: {default_policy}\n")
    print(f"{'RFQ':<10} {'TUD':<10} {'Predicted Price':<20}")
    print("-" * 40)
    for rfq_val, tud_val in test_cases:
        price = predict_price(rfq_val, tud_val)
        print(f"{rfq_val:<10.1f} {tud_val:<10.1f} ${price:<19.2f}")

    print("\n" + "=" * 70)
    print("Available State Variables:")
    print("=" * 70)
    print("  RFQ - Request for Quote")
    print("  TUD - Time Until Delivery")

    print("\nAvailable Operations:")
    print("  Binary: add, subtract, multiply, protected_div, maximum, minimum")
    print("  Unary: protected_sqrt, square, lf (logistic function)")
    print("=" * 70)


if __name__ == "__main__":
    main()