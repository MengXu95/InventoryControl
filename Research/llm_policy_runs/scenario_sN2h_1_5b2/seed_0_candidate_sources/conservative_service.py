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
