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