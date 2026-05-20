def replenishment_policy(site_state, global_state):
    system_pressure = global_state['total_forecast'] - global_state['total_inventory']
    local_target = site_state['forecast_1'] + site_state['forecast_2'] - site_state['pipeline']
    if system_pressure > 0:
        local_target = local_target + system_pressure / max(1, global_state['num_sites'])
    gap = local_target - site_state['inventory_level']
    return max(0, min(site_state['production_capacity'], gap))


def transshipment_proposal(source_state, target_state, global_state):
    source_surplus = source_state['inventory_level'] - source_state['forecast_1'] - source_state['forecast_2']
    target_shortage = target_state['forecast_1'] + target_state['forecast_2'] - target_state['inventory_level']
    forward = min(max(0, source_surplus), max(0, target_shortage))
    reverse_surplus = target_state['inventory_level'] - target_state['forecast_1'] - target_state['forecast_2']
    reverse_shortage = source_state['forecast_1'] + source_state['forecast_2'] - source_state['inventory_level']
    reverse = min(max(0, reverse_surplus), max(0, reverse_shortage))
    if global_state['inventory_imbalance'] < max(1, global_state['average_inventory'] * 0.25):
        return 0
    return forward - reverse


def consensus_gate(source_state, target_state, proposed_quantity, history):
    donor_margin = source_state['inventory_level'] - source_state['forecast_1'] - source_state['forecast_2']
    receiver_gap = target_state['forecast_1'] + target_state['forecast_2'] - target_state['inventory_level']
    if donor_margin <= 0 or receiver_gap <= 0:
        return 0
    fairness = max(0, 1 - abs(history['net_pair_transfer']) / max(1, source_state['capacity']))
    transfer_limit = min(donor_margin * 0.75, receiver_gap)
    return min(proposed_quantity, transfer_limit) * fairness
