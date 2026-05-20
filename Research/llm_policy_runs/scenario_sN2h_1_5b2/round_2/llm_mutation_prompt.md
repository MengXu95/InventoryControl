# ConsensusCode-Evolve Mutation Prompt: round 2

You are improving Python inventory-control policies for a multi-site replenishment/transshipment simulator.
Generate exactly 3 new candidate policy files. Keep each policy short and deterministic.

## Objective
- Minimize fitness: average total cost = transshipment + holding/lost-sales + order cost.
- Improve consensus-seeking: avoid draining a donor site and avoid unfair repeated donor behavior.
- Prefer robust service behavior without excessive orders.

## Allowed API
```python
def replenishment_policy(site_state, global_state):
    return order_quantity

def transshipment_proposal(source_state, target_state, global_state):
    return proposed_quantity

def consensus_gate(source_state, target_state, proposed_quantity, history):
    return accepted_quantity
```

## Available State Keys
site_state keys:
- site_index, inventory_level, holding_cost, lost_sales_cost, capacity
- production_capacity, fixed_order_cost, per_unit_order_cost, pipeline
- forecast_1, forecast_2, transshipment_cost, fixed_transshipment_cost

global_state keys:
- period, episode_length, num_sites, total_inventory, total_forecast
- average_inventory, inventory_imbalance, forecast_imbalance

history keys:
- period, net_pair_transfer, accepted_transfer_count, rejected_transfer_count

## Safety Rules
- No imports.
- No file, network, random, object attribute, class, list, or dict construction.
- Only arithmetic, if statements, comparisons, and calls to abs, min, max, round.
- Return nonnegative quantities from replenishment_policy and consensus_gate.
- transshipment_proposal may return positive or negative signed quantities.

## Scenario And Seed
- dataset: sN2h_1_5b2
- seed: 0

## Current Top Policy Scores
| policy | fitness | trans | hold_lost | order | stockout_proxy | accepted | rejected | fairness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conservative_service | 126.3831 | 3.0672 | 77.7795 | 45.5364 | 27 | 16 | 0 | 25.75 |
| no_transship_reorder | 143.5111 | 0.0000 | 100.9070 | 42.6041 | 25 | 0 | 0 | 0.00 |
| lean_emergency_transfer | 147.7086 | 2.4573 | 105.0769 | 40.1744 | 31 | 10 | 0 | 0.07 |

## Current Top Policy Sources
### conservative_service
```python
def replenishment_policy(site_state, global_state):
    service_target = site_state['forecast_1'] + site_state['forecast_2'] + site_state['lost_sales_cost']
    gap = service_target - site_state['inventory_level'] - site_state['pipeline']
    return max(0, min(site_state['production_capacity'] * 1.5, gap))


def transshipment_proposal(source_state, target_state, global_state):
    source_buffer = source_state['inventory_level'] - source_state['forecast_1'] - source_state['forecast_2']
    target_need = target_state['forecast_1'] + target_state['forecast_2'] - target_state['inventory_level']
    transfer = min(max(0, source_buffer), max(0, target_need))
    reverse_buffer = target_state['inventory_level'] - target_state['forecast_1'] - target_state['forecast_2']
    reverse_need = source_state['forecast_1'] + source_state['forecast_2'] - source_state['inventory_level']
    reverse_transfer = min(max(0, reverse_buffer), max(0, reverse_need))
    return transfer - reverse_transfer


def consensus_gate(source_state, target_state, proposed_quantity, history):
    donor_safe_inventory = source_state['inventory_level'] - source_state['forecast_1'] - source_state['forecast_2']
    receiver_gap = target_state['forecast_1'] + target_state['forecast_2'] - target_state['inventory_level']
    if donor_safe_inventory <= 0 or receiver_gap <= 0:
        return 0
    fairness_factor = max(0, 1 - abs(history['net_pair_transfer']) / max(1, source_state['capacity'] * 0.5))
    return min(proposed_quantity, donor_safe_inventory, receiver_gap) * fairness_factor
```

### no_transship_reorder
```python
def replenishment_policy(site_state, global_state):
    target = site_state['forecast_1'] + site_state['forecast_2']
    return max(0, min(site_state['production_capacity'], target - site_state['inventory_level']))


def transshipment_proposal(source_state, target_state, global_state):
    return 0


def consensus_gate(source_state, target_state, proposed_quantity, history):
    return 0
```

### lean_emergency_transfer
```python
def replenishment_policy(site_state, global_state):
    lean_target = site_state['forecast_1'] + site_state['pipeline']
    if site_state['lost_sales_cost'] > site_state['holding_cost'] * 5:
        lean_target = lean_target + site_state['forecast_2'] * 0.5
    gap = lean_target - site_state['inventory_level']
    return max(0, min(site_state['production_capacity'] * 0.9, gap))


def transshipment_proposal(source_state, target_state, global_state):
    source_surplus = source_state['inventory_level'] - source_state['forecast_1'] - source_state['forecast_2']
    target_emergency = target_state['forecast_1'] - target_state['inventory_level']
    forward = min(max(0, source_surplus), max(0, target_emergency))
    reverse_surplus = target_state['inventory_level'] - target_state['forecast_1'] - target_state['forecast_2']
    reverse_emergency = source_state['forecast_1'] - source_state['inventory_level']
    reverse = min(max(0, reverse_surplus), max(0, reverse_emergency))
    return forward - reverse


def consensus_gate(source_state, target_state, proposed_quantity, history):
    donor_safe = source_state['inventory_level'] - source_state['forecast_1'] - source_state['forecast_2']
    receiver_emergency = target_state['forecast_1'] - target_state['inventory_level']
    if donor_safe <= 0:
        return 0
    if receiver_emergency <= 0:
        return 0
    fairness = max(0, 1 - abs(history['net_pair_transfer']) / max(1, source_state['capacity'] * 0.5))
    return min(proposed_quantity, donor_safe, receiver_emergency) * fairness
```

## Required Output Format
Return exactly three fenced Python blocks. Start each block with a filename comment:

```python
# filename: candidate_1.py
def replenishment_policy(site_state, global_state):
    ...
```

Suggested mutation themes:
1. conservative donor protection with strong service recovery,
2. fairness-aware consensus with moderate transfer frequency,
3. lean ordering with emergency-only transshipment.

After receiving the LLM response, save the full response as `llm_response.md` in this round folder, or save each candidate `.py` file directly into `generated_candidates/`.