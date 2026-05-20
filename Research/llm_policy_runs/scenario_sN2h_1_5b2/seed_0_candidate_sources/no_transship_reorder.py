def replenishment_policy(site_state, global_state):
    target = site_state['forecast_1'] + site_state['forecast_2']
    return max(0, min(site_state['production_capacity'], target - site_state['inventory_level']))


def transshipment_proposal(source_state, target_state, global_state):
    return 0


def consensus_gate(source_state, target_state, proposed_quantity, history):
    return 0
