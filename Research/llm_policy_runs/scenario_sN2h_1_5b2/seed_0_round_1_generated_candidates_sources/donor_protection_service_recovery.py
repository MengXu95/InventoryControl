def replenishment_policy(site_state, global_state):
    near_term_need = site_state['forecast_1'] + site_state['forecast_2']
    risk_buffer = site_state['lost_sales_cost'] / max(1, site_state['holding_cost'])
    target = near_term_need + risk_buffer + site_state['pipeline']
    gap = target - site_state['inventory_level']
    return max(0, min(site_state['production_capacity'] * 1.25, gap))


def transshipment_proposal(source_state, target_state, global_state):
    source_safe = source_state['inventory_level'] - source_state['forecast_1'] - source_state['pipeline']
    target_gap = target_state['forecast_1'] + target_state['pipeline'] - target_state['inventory_level']
    forward = min(max(0, source_safe), max(0, target_gap))
    reverse_safe = target_state['inventory_level'] - target_state['forecast_1'] - target_state['pipeline']
    reverse_gap = source_state['forecast_1'] + source_state['pipeline'] - source_state['inventory_level']
    reverse = min(max(0, reverse_safe), max(0, reverse_gap))
    return forward - reverse


def consensus_gate(source_state, target_state, proposed_quantity, history):
    donor_buffer = source_state['inventory_level'] - source_state['forecast_1'] - source_state['pipeline']
    receiver_need = target_state['forecast_1'] + target_state['pipeline'] - target_state['inventory_level']
    if donor_buffer <= source_state['forecast_2'] * 0.5:
        return 0
    if receiver_need <= 0:
        return 0
    fairness = max(0, 1 - abs(history['net_pair_transfer']) / max(1, source_state['capacity'] * 0.75))
    urgency = min(1, receiver_need / max(1, target_state['forecast_1'] + target_state['forecast_2']))
    return min(proposed_quantity, donor_buffer, receiver_need) * max(0.25, urgency) * fairness
