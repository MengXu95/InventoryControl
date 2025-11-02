# Policy Optimization Modules

## Overview

This package provides three modules for optimizing business decisions using genetic programming-based policy expressions:

1. **Replenishment Policy Module** (`replenishment_policy.py`) - Determines optimal order quantities for individual retailers
2. **Transshipment Policy Module** (`transshipment_policy.py`) - Determines optimal transfer quantities between retailer pairs
3. **Price Prediction Policy Module** (`price_predict_policy.py`) - Predicts prices based on request for quote (RFQ) and time until delivery (TUD)

All modules use expression tree evaluation to calculate decisions based on state variables and support complex mathematical operations with overflow protection.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module 1: Replenishment Policy](#module-1-replenishment-policy)
- [Module 2: Transshipment Policy](#module-2-transshipment-policy)
- [Module 3: Price Prediction Policy](#module-3-price-prediction-policy)
- [Policy Expression Syntax](#policy-expression-syntax)
- [Available Variables](#available-variables)
- [Available Operations](#available-operations)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Installation

### Requirements

- Python 3.7 or higher
- NumPy

### Setup

```bash
# Install required package
pip install numpy

# Download the modules
# - replenishment_policy.py
# - transshipment_policy.py
# - price_predict_policy.py
```

---

## Quick Start

### Running the Demos

All modules include demonstration scripts:

```bash
# Run replenishment policy demo
python replenishment_policy.py

# Run transshipment policy demo
python transshipment_policy.py

# Run price prediction policy demo
python price_predict_policy.py
```

### Basic Usage in Your Code

**Replenishment Policy:**
```python
from replenishment_policy import calculate_replenishment_quantity

# Calculate order quantity for a retailer
quantity = calculate_replenishment_quantity(
    my_retailer,
    policy_expression="subtract('FC1', 'INL')"
)
print(f"Order {quantity} units")
```

**Transshipment Policy:**
```python
from transshipment_policy import calculate_transshipment_quantity

# Calculate transshipment between two retailers
quantity = calculate_transshipment_quantity(
    retailer_a,
    retailer_b,
    policy_expression="subtract('INL1', 'INL2')"
)
print(f"Transfer {quantity} units")
```

**Price Prediction Policy:**
```python
from price_predict_policy import predict_price

# Predict price based on RFQ and TUD
price = predict_price(
    rfq=100.0,
    tud=5.0,
    policy_expression="protected_div('RFQ', 'TUD')"
)
print(f"Predicted price: ${price:.2f}")
```

---

## Module 1: Replenishment Policy

### Purpose

Determines how much inventory a single retailer should order from their supplier.

### Required Retailer Attributes

Your retailer object must have these attributes:

```python
class Retailer:
    inv_level                        # Current inventory level (float)
    holding_cost                     # Cost per unit held (float)
    lost_sales_cost                  # Cost per unit of lost sales (float)
    capacity                         # Maximum inventory capacity (float)
    fixed_order_cost                 # Fixed cost per order (float)
    pipeline                         # List of orders in transit, e.g., [30.0, 20.0]
    forecast                         # Demand forecast list, e.g., [80.0, 75.0]
    transshipment_cost               # Cost per unit transshipped (float)
    fixed_order_transshipment_cost   # Fixed transshipment cost (float)
```

### Function Signature

```python
calculate_replenishment_quantity(retailer, policy_expression=None)
```

**Parameters:**
- `retailer`: Retailer object with required attributes
- `policy_expression`: String or list representing the policy (optional)
  - If `None`, uses a default policy
  - Can be a string: `"subtract('FC1', 'INL')"`
  - Can be a list: `['subtract', 'FC1', 'INL']`

**Returns:**
- `float`: Recommended order quantity, rounded to nearest integer

### Example

```python
from replenishment_policy import calculate_replenishment_quantity, Retailer

# Create a retailer
retailer = Retailer(
    inv_level=50.0,
    holding_cost=2.5,
    lost_sales_cost=15.0,
    capacity=200.0,
    fixed_order_cost=100.0,
    pipeline=[30.0, 20.0],
    forecast=[80.0, 75.0],
    transshipment_cost=5.0,
    fixed_order_transshipment_cost=50.0
)

# Simple policy: order the difference between forecast and current inventory
policy = "subtract('FC1', 'INL')"
order_qty = calculate_replenishment_quantity(retailer, policy)

print(f"Recommended order quantity: {order_qty} units")
# Output: Recommended order quantity: 30.0 units
```

---

## Module 2: Transshipment Policy

### Purpose

Determines how much inventory should be transferred between two retailers to balance stock levels.

### Required Retailer Attributes

Both retailer objects must have these attributes:

```python
class Retailer:
    inv_level                        # Current inventory level (float)
    holding_cost                     # Cost per unit held (float)
    lost_sales_cost                  # Cost per unit of lost sales (float)
    capacity                         # Maximum inventory capacity (float)
    fixed_order_cost                 # Fixed cost per order (float)
    pipeline                         # List of orders in transit
    forecast                         # Demand forecast list
    transshipment_cost               # Cost per unit transshipped (float)
    fixed_order_transshipment_cost   # Fixed transshipment cost (float)
```

### Function Signature

```python
calculate_transshipment_quantity(retailer_i, retailer_j, 
                                  retailer_i_id=0, retailer_j_id=1,
                                  policy_expression=None)
```

**Parameters:**
- `retailer_i`: First retailer object (source)
- `retailer_j`: Second retailer object (destination)
- `retailer_i_id`: Optional identifier for retailer i (default: 0)
- `retailer_j_id`: Optional identifier for retailer j (default: 1)
- `policy_expression`: String or list representing the policy (optional)

**Returns:**
- `float`: Recommended transshipment quantity
  - **Positive**: Ship from retailer_i to retailer_j
  - **Negative**: Ship from retailer_j to retailer_i
  - **Zero**: No transshipment needed

### Example

```python
from transshipment_policy import calculate_transshipment_quantity, Retailer

# Create two retailers with different inventory levels
retailer_a = Retailer(
    inv_level=80.0,  # High inventory
    holding_cost=2.0,
    lost_sales_cost=12.0,
    capacity=200.0,
    fixed_order_cost=100.0,
    pipeline=[20.0, 15.0],
    forecast=[60.0, 55.0],
    transshipment_cost=3.0,
    fixed_order_transshipment_cost=40.0
)

retailer_b = Retailer(
    inv_level=25.0,  # Low inventory
    holding_cost=2.5,
    lost_sales_cost=18.0,
    capacity=180.0,
    fixed_order_cost=100.0,
    pipeline=[10.0, 12.0],
    forecast=[90.0, 85.0],
    transshipment_cost=3.0,
    fixed_order_transshipment_cost=40.0
)

# Simple policy: ship based on inventory difference
policy = "subtract('INL1', 'INL2')"
transship_qty = calculate_transshipment_quantity(retailer_a, retailer_b, policy_expression=policy)

print(f"Recommended transshipment: {transship_qty} units")
# Output: Recommended transshipment: 55.0 units

if transship_qty > 0:
    print(f"Ship {transship_qty} units from A to B")
elif transship_qty < 0:
    print(f"Ship {abs(transship_qty)} units from B to A")
else:
    print("No transshipment needed")
```

---

## Module 3: Price Prediction Policy

### Purpose

Predicts prices based on Request for Quote (RFQ) and Time Until Delivery (TUD) using genetic programming-based policy expressions.

### Input Requirements

The function requires two simple numeric inputs:

- `rfq` (float): Request for Quote value
- `tud` (float): Time Until Delivery value

### Function Signature

```python
predict_price(rfq, tud, policy_expression=None)
```

**Parameters:**
- `rfq`: Request for Quote value (float)
- `tud`: Time Until Delivery value (float)
- `policy_expression`: String or list representing the policy (optional)
  - If `None`, uses default policy: `"protected_div('RFQ', 'TUD')"`
  - Can be a string: `"protected_div('RFQ', 'TUD')"`
  - Can be a list: `['protected_div', 'RFQ', 'TUD']`

**Returns:**
- `float`: Predicted price, rounded to 2 decimal places

### Example

```python
from price_predict_policy import predict_price

# Simple case: price is RFQ divided by TUD
rfq_value = 100.0
tud_value = 5.0

# Use default policy
predicted_price = predict_price(rfq_value, tud_value)
print(f"Predicted price: ${predicted_price:.2f}")
# Output: Predicted price: $20.00

# Custom policy: weighted combination
custom_policy = "add(protected_div('RFQ', 'TUD'), multiply('RFQ', 0.1))"
custom_price = predict_price(rfq_value, tud_value, policy_expression=custom_policy)
print(f"Custom predicted price: ${custom_price:.2f}")
```

### Batch Prediction Example

```python
from price_predict_policy import predict_price

# Multiple RFQ-TUD pairs
quotes = [
    (150.0, 10.0),
    (80.0, 4.0),
    (200.0, 8.0),
    (50.0, 2.5)
]

policy = "protected_div('RFQ', 'TUD')"

print(f"{'RFQ':<10} {'TUD':<10} {'Predicted Price':<20}")
print("-" * 40)
for rfq, tud in quotes:
    price = predict_price(rfq, tud, policy_expression=policy)
    print(f"{rfq:<10.1f} {tud:<10.1f} ${price:<19.2f}")

# Output:
# RFQ        TUD        Predicted Price     
# ----------------------------------------
# 150.0      10.0       $15.00              
# 80.0       4.0        $20.00              
# 200.0      8.0        $25.00              
# 50.0       2.5        $20.00
```

---

## Policy Expression Syntax

Policies can be written as:
1. **String expressions** (recommended for readability)
2. **List expressions** (used internally)

### String Format

Use nested function notation with quoted variables:

```python
# Simple examples
"subtract('FC1', 'INL')"
"maximum('FC1', 'FC2')"
"add('INL', 'PIP')"
"protected_div('RFQ', 'TUD')"

# Complex nested examples
"subtract(add('FC1', 'FC2'), add('INL', 'PIP'))"
"multiply(protected_div('INL1', 'INL2'), 'PTC')"
"square(multiply('PLSC2', 'PHC1'))"
"add(protected_div('RFQ', 'TUD'), lf('TUD'))"
```

### List Format

Flat list representation (automatically converted from strings):

```python
# Equivalent to "subtract('FC1', 'INL')"
['subtract', 'FC1', 'INL']

# Equivalent to "add(subtract('FC1', 'INL'), 'PIP')"
['add', 'subtract', 'FC1', 'INL', 'PIP']

# Equivalent to "protected_div('RFQ', 'TUD')"
['protected_div', 'RFQ', 'TUD']
```

### Converting Between Formats

```python
from replenishment_policy import parse_policy_string

# Convert string to list
policy_str = "subtract('FC1', 'INL')"
policy_list = parse_policy_string(policy_str)
print(policy_list)  # Output: ['subtract', 'FC1', 'INL']
```

---

## Available Variables

### Replenishment Policy Variables

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `INL` | Inventory Level | 50.0 |
| `PHC` | Per-unit Holding Cost | 2.5 |
| `PLSC` | Per-unit Lost Sales Cost | 15.0 |
| `INC` | Inventory Capacity | 200.0 |
| `FOC` | Fixed Order Cost | 100.0 |
| `PIP` | Pipeline inventory (next delivery) | 30.0 |
| `FC1` | Forecast period 1 | 80.0 |
| `FC2` | Forecast period 2 | 75.0 |
| `PTC` | Per-unit Transshipment Cost | 5.0 |
| `FTC` | Fixed Transshipment Cost | 50.0 |

### Transshipment Policy Variables

Variables are numbered 1 and 2 for each retailer:

**Retailer 1 (i) Variables:**

| Variable | Description |
|----------|-------------|
| `INL1` | Inventory Level - Retailer 1 |
| `PHC1` | Per-unit Holding Cost - Retailer 1 |
| `PLSC1` | Per-unit Lost Sales Cost - Retailer 1 |
| `INC1` | Inventory Capacity - Retailer 1 |
| `FOC1` | Fixed Order Cost - Retailer 1 |
| `PIP1` | Pipeline inventory - Retailer 1 |
| `FC11` | Forecast period 1 - Retailer 1 |
| `FC12` | Forecast period 2 - Retailer 1 |

**Retailer 2 (j) Variables:**

| Variable | Description |
|----------|-------------|
| `INL2` | Inventory Level - Retailer 2 |
| `PHC2` | Per-unit Holding Cost - Retailer 2 |
| `PLSC2` | Per-unit Lost Sales Cost - Retailer 2 |
| `INC2` | Inventory Capacity - Retailer 2 |
| `FOC2` | Fixed Order Cost - Retailer 2 |
| `PIP2` | Pipeline inventory - Retailer 2 |
| `FC21` | Forecast period 1 - Retailer 2 |
| `FC22` | Forecast period 2 - Retailer 2 |

**Shared Variables:**

| Variable | Description |
|----------|-------------|
| `PTC` | Per-unit Transshipment Cost |
| `FTC` | Fixed Transshipment Cost |

### Price Prediction Policy Variables

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `RFQ` | Request for Quote | 100.0 |
| `TUD` | Time Until Delivery | 5.0 |

---

## Available Operations

### Binary Operations (require 2 inputs)

| Operation | Description | Example |
|-----------|-------------|---------|
| `add` | Addition | `add('FC1', 'FC2')` |
| `subtract` | Subtraction | `subtract('FC1', 'INL')` |
| `multiply` | Multiplication | `multiply('PHC', 'INL')` |
| `protected_div` | Division (safe, returns 1 if division by zero) | `protected_div('RFQ', 'TUD')` |
| `maximum` | Maximum of two values | `maximum('FC1', 'FC2')` |
| `minimum` | Minimum of two values | `minimum('INL', 'INC')` |

### Unary Operations (require 1 input)

| Operation | Description | Example |
|-----------|-------------|---------|
| `protected_sqrt` | Square root (returns 0 for negative) | `protected_sqrt('INL')` |
| `square` | Square of a value | `square('PHC')` |
| `lf` | Logistic/sigmoid function: 1/(1+e^-x) | `lf('TUD')` |

### Safety Features

All operations include overflow and underflow protection:
- Division by zero returns 1
- Square root of negative numbers returns 0
- Overflow/underflow results return infinity
- Invalid operations return safe default values

---

## Examples

### Replenishment Policy Examples

**Example 1: Simple Forecast-Based Replenishment**

Order based on the difference between next period's forecast and current inventory:

```python
policy = "subtract('FC1', 'INL')"
# If FC1 = 80 and INL = 50, order 30 units
```

**Example 2: Economic Order Quantity Approximation**

Use holding and lost sales costs to determine order quantity:

```python
policy = "multiply(protected_sqrt(protected_div('PLSC', 'PHC')), 'FC1')"
# Orders more when lost sales cost is high relative to holding cost
```

### Transshipment Policy Examples

**Example 3: Transshipment Based on Inventory Imbalance**

Transfer inventory from high-stock to low-stock retailer:

```python
policy = "subtract('INL1', 'INL2')"
# If INL1 = 80 and INL2 = 25, ship 55 units from retailer 1 to 2
```

**Example 4: Demand-Aware Transshipment**

Consider both inventory levels and forecasts:

```python
policy = "subtract(subtract('INL1', 'FC11'), subtract('INL2', 'FC21'))"
# Ships from retailer with excess to retailer with deficit
```

### Price Prediction Policy Examples

**Example 5: Simple Linear Price Model**

Price inversely proportional to delivery time:

```python
policy = "protected_div('RFQ', 'TUD')"
# If RFQ = 100 and TUD = 5, price = 20
```

**Example 6: Premium for Fast Delivery**

Add premium when TUD is low:

```python
policy = "add('RFQ', protected_div('RFQ', multiply('TUD', 'TUD')))"
# Adds exponential premium for shorter delivery times
```

**Example 7: Sigmoid-Scaled Price**

Use logistic function to scale prices smoothly:

```python
policy = "multiply('RFQ', lf(protected_div('RFQ', 'TUD')))"
# Applies sigmoid transformation for smooth price curves
```

### Complex Multi-Factor Policy

**Example 8: Sophisticated Replenishment Decision**

Combine multiple factors for sophisticated decision-making:

```python
policy = """
multiply(
    subtract('FC1', add('INL', 'PIP')),
    protected_div('PLSC', add('PHC', 'FOC'))
)
"""
# Considers forecast, current inventory, pipeline, and all costs
```

---

## Troubleshooting

### Common Issues

**1. AttributeError: 'Retailer' object has no attribute 'pipeline'**

Solution: Ensure your retailer object has all required attributes. The `pipeline` and `forecast` must be lists.

```python
# Correct
retailer.pipeline = [30.0, 20.0]
retailer.forecast = [80.0, 75.0]

# Incorrect
retailer.pipeline = 30.0  # Should be a list
```

**2. ValueError: Unknown node type**

Solution: Check that all variable names in your policy are spelled correctly and exist in the available variables list for the specific module you're using.

```python
# Correct (Replenishment)
policy = "subtract('FC1', 'INL')"

# Correct (Price Prediction)
policy = "protected_div('RFQ', 'TUD')"

# Incorrect
policy = "subtract('FORECAST1', 'INVENTORY')"  # Wrong variable names
```

**3. IndexError: list index out of range**

Solution: For replenishment/transshipment policies, ensure `pipeline` and `forecast` lists have at least 2 elements (indices 0 and 1).

```python
# Correct
retailer.pipeline = [30.0, 20.0]  # Length >= 1
retailer.forecast = [80.0, 75.0]  # Length >= 2

# Incorrect
retailer.pipeline = []  # Too short
retailer.forecast = [80.0]  # Too short
```

**4. Policy returns unexpected values**

Solution: Test your policy step by step using the demo functions. Print intermediate values:

```python
# Debug replenishment policy
simple_policy = "subtract('FC1', 'INL')"
result = calculate_replenishment_quantity(retailer, simple_policy)
print(f"FC1={retailer.forecast[0]}, INL={retailer.inv_level}, Result={result}")

# Debug price prediction policy
simple_policy = "protected_div('RFQ', 'TUD')"
result = predict_price(100.0, 5.0, simple_policy)
print(f"RFQ=100.0, TUD=5.0, Result={result}")
```

### Module-Specific Troubleshooting

**Price Prediction Module:**
- Ensure RFQ and TUD are numeric (float or int)
- Check for division by zero in your policy (use `protected_div`)
- Verify that variable names are exactly `'RFQ'` and `'TUD'` (case-sensitive)

**Replenishment Module:**
- Verify all 9 retailer attributes are present
- Ensure `pipeline[0]` and `forecast[0]`, `forecast[1]` are accessible
- Check that costs are positive numbers

**Transshipment Module:**
- Both retailers must have identical attribute structures
- Check that retailer IDs are correctly specified
- Verify transshipment cost variables are present

### Getting Help

1. Run the demo scripts to see working examples:
   ```bash
   python replenishment_policy.py
   python transshipment_policy.py
   python price_predict_policy.py
   ```

2. Check the function docstrings:
   ```python
   from price_predict_policy import predict_price
   help(predict_price)
   ```

3. Verify your inputs match the required format:
   ```python
   # For price prediction
   assert isinstance(rfq, (int, float)), "RFQ must be numeric"
   assert isinstance(tud, (int, float)), "TUD must be numeric"
   assert tud != 0, "TUD should not be zero to avoid division issues"
   ```

---

## Best Practices

### 1. Start Simple

Begin with simple policies and gradually increase complexity:

```python
# Start here
"subtract('FC1', 'INL')"
"protected_div('RFQ', 'TUD')"

# Then try
"subtract(add('FC1', 'FC2'), add('INL', 'PIP'))"
"add('RFQ', multiply('TUD', 'TUD'))"

# Finally
"multiply(subtract(add('FC1', 'FC2'), add('INL', 'PIP')), 
          protected_div('PLSC', 'PHC'))"
```

### 2. Test Policies with Known Data

Create test cases with known values to verify policy behavior:

```python
# Test replenishment
test_retailer = Retailer(
    inv_level=50.0,
    holding_cost=2.0,
    lost_sales_cost=10.0,
    capacity=100.0,
    fixed_order_cost=50.0,
    pipeline=[10.0, 10.0],
    forecast=[60.0, 60.0],
    transshipment_cost=3.0,
    fixed_order_transshipment_cost=25.0
)

# Expected result: 60 - 50 = 10
result = calculate_replenishment_quantity(
    test_retailer, 
    "subtract('FC1', 'INL')"
)
assert result == 10.0, f"Expected 10.0, got {result}"

# Test price prediction
# Expected result: 100 / 5 = 20
price = predict_price(100.0, 5.0, "protected_div('RFQ', 'TUD')")
assert price == 20.0, f"Expected 20.0, got {price}"
```

### 3. Document Your Policies

Add comments explaining your policy logic:

```python
# Inventory Position Policy:
# Order enough to reach forecast level considering
# current inventory and pipeline
policy = "subtract('FC1', add('INL', 'PIP'))"

# Cost-Weighted Transshipment:
# Ship more when receiving retailer has high lost sales cost
policy = "multiply(subtract('INL1', 'INL2'), 'PLSC2')"

# Urgency-Based Pricing:
# Higher prices for faster delivery requirements
policy = "multiply('RFQ', protected_div(10, 'TUD'))"
```

### 4. Handle Edge Cases

Consider adding bounds or using minimum/maximum operators:

```python
# Never order more than capacity allows
policy = "minimum(subtract('FC1', 'INL'), 'INC')"

# Ensure price has a minimum floor
policy = "maximum(protected_div('RFQ', 'TUD'), 5.0)"
# Note: Numeric literals would need to be handled separately in actual implementation
```

### 5. Use Appropriate Rounding

Be aware of rounding behavior:
- Replenishment quantities: Rounded to nearest integer
- Transshipment quantities: Rounded to 2 decimal places
- Prices: Rounded to 2 decimal places

---

## Advanced Usage

### Creating Custom Policies from Genetic Programming

If you're using genetic programming to evolve policies, all modules accept policy expressions directly:

```python
# Evolved policy from GP algorithm
evolved_policy = "square(multiply('PLSC2', multiply('PHC1', protected_div('FOC1', multiply('FTC', 'FTC')))))"

# Use it directly
quantity = calculate_transshipment_quantity(
    retailer_a, 
    retailer_b, 
    policy_expression=evolved_policy
)

# Price prediction with evolved policy
price_policy = "multiply(lf(protected_div('RFQ', 'TUD')), add('RFQ', square('TUD')))"
price = predict_price(rfq, tud, policy_expression=price_policy)
```

### Batch Processing

**Process Multiple Retailers:**
```python
retailers = [retailer1, retailer2, retailer3, retailer4]
policy = "subtract('FC1', 'INL')"

order_quantities = []
for retailer in retailers:
    qty = calculate_replenishment_quantity(retailer, policy)
    order_quantities.append(qty)
    print(f"Retailer {retailer.id}: Order {qty} units")
```

**Process Multiple Price Quotes:**
```python
quotes = [(100, 5), (150, 10), (80, 4), (200, 8)]
policy = "protected_div('RFQ', 'TUD')"

prices = [predict_price(rfq, tud, policy) for rfq, tud in quotes]
print("Predicted prices:", prices)
```

### Integration with Simulation Systems

All modules are designed to integrate with discrete-event simulation systems:

```python
import simpy

def retailer_process(env, retailer, policy):
    while True:
        # Calculate order quantity
        order_qty = calculate_replenishment_quantity(retailer, policy)
        
        # Place order in simulation
        if order_qty > 0:
            place_order(retailer, order_qty)
        
        # Wait for next review period
        yield env.timeout(1)

# Setup simulation
env = simpy.Environment()
env.process(retailer_process(env, my_retailer, my_policy))
env.run(until=100)
```

### Price Prediction in Real-Time Systems

```python
def dynamic_pricing_engine(rfq_stream, tud_stream, policy):
    """
    Real-time price prediction for incoming quotes
    """
    prices = []
    for rfq, tud in zip(rfq_stream, tud_stream):
        price = predict_price(rfq, tud, policy)
        prices.append(price)
        # Apply business rules
        if price < minimum_acceptable_price:
            price = minimum_acceptable_price
        yield price
```

### Cross-Module Integration

Combine multiple modules for comprehensive optimization:

```python
# First, predict the price
predicted_price = predict_price(rfq, tud, price_policy)

# Then, use price to influence replenishment decision
# (Note: This requires custom integration as modules use different variables)
order_qty = calculate_replenishment_quantity(retailer, replenishment_policy)

# Consider transshipment opportunities
for other_retailer in network:
    transship_qty = calculate_transshipment_quantity(
        retailer, 
        other_retailer, 
        transshipment_policy
    )
    if transship_qty > threshold:
        execute_transshipment(retailer, other_retailer, transship_qty)
```

---

## Performance Considerations

### Policy Complexity

More complex policies take longer to evaluate:
- Simple policies (1-3 operations): ~0.001ms per evaluation
- Medium policies (4-10 operations): ~0.01ms per evaluation  
- Complex policies (>10 operations): ~0.1ms per evaluation

### Optimization Tips

1. **Reuse parsed policies**: Parse once, use many times
```python
from replenishment_policy import parse_policy_string

# Parse once
policy_list = parse_policy_string("subtract('FC1', 'INL')")

# Use many times
for retailer in retailers:
    qty = calculate_replenishment_quantity(retailer, policy_list)
```

2. **Vectorize when possible**: For batch predictions with same policy
```python
# Instead of loop
prices = [predict_price(rfq, tud, policy) for rfq, tud in data]

# Consider numpy arrays for custom batch operations
import numpy as np
rfqs = np.array([d[0] for d in data])
tuds = np.array([d[1] for d in data])
# Then process in batches
```

3. **Profile complex policies**: Use Python's profiling tools
```python
import cProfile
cProfile.run('calculate_replenishment_quantity(retailer, complex_policy)')
```

---

## License and Citation

These modules are provided for research and educational purposes. If you use these modules in your research, please cite appropriately.

## Support

For questions, issues, or contributions, please refer to the module docstrings and inline comments for detailed implementation information.

---

**Module Versions:**
- Replenishment Policy: 1.0.0
- Transshipment Policy: 1.0.0
- Price Prediction Policy: 1.0.0

**Last Updated:** November 2025