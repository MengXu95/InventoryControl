def replenishment_policy(site_state, global_state):
    target = site_state['forecast_1'] + site_state['forecast_2'] + site_state['pipeline']
    shortage_gap = target - site_state['inventory_level']
    capacity_gate = site_state['production_capacity']
    return max(0, min(capacity_gate, shortage_gap))


def transshipment_proposal(source_state, target_state, global_state):
    source_pressure = source_state['inventory_level'] - source_state['forecast_1'] - source_state['pipeline']
    target_pressure = target_state['forecast_1'] + target_state['pipeline'] - target_state['inventory_level']
    transfer = min(max(0, source_pressure), max(0, target_pressure))
    reverse_source_pressure = target_state['inventory_level'] - target_state['forecast_1'] - target_state['pipeline']
    reverse_target_pressure = source_state['forecast_1'] + source_state['pipeline'] - source_state['inventory_level']
    reverse_transfer = min(max(0, reverse_source_pressure), max(0, reverse_target_pressure))
    return transfer - reverse_transfer


def consensus_gate(source_state, target_state, proposed_quantity, history):
    donor_buffer = source_state['inventory_level'] - source_state['forecast_1'] - source_state['pipeline']
    receiver_gap = target_state['forecast_1'] + target_state['pipeline'] - target_state['inventory_level']
    donor_willingness = min(1, max(0, donor_buffer / max(1, source_state['capacity'])))
    receiver_urgency = min(1, max(0, receiver_gap / max(1, target_state['capacity'])))
    fairness_factor = max(0, 1 - abs(history['net_pair_transfer']) / max(1, source_state['capacity']))
    return proposed_quantity * min(donor_willingness + receiver_urgency, 1) * fairness_factor
