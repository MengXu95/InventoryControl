import numpy as np

from MTGP_niching_replenish_transship.Inventory_simulator_replenish_transship import InvOptEnv


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def replenishment_state_to_dict(state, site_index):
    return {
        'site_index': site_index,
        'inventory_level': float(state[0]),
        'holding_cost': float(state[1]),
        'lost_sales_cost': float(state[2]),
        'capacity': float(state[3]),
        'production_capacity': float(state[4]),
        'fixed_order_cost': float(state[5]),
        'per_unit_order_cost': float(state[6]),
        'pipeline': float(state[7]),
        'forecast_1': float(state[8]),
        'forecast_2': float(state[9]),
        'transshipment_cost': float(state[10]),
        'fixed_transshipment_cost': float(state[11]),
    }


def transshipment_state_to_pair(state):
    site_i = {
        'site_index': int(state[0]),
        'inventory_level': float(state[2]),
        'holding_cost': float(state[3]),
        'lost_sales_cost': float(state[4]),
        'capacity': float(state[5]),
        'fixed_order_cost': float(state[6]),
        'pipeline': float(state[7]),
        'forecast_1': float(state[8]),
        'forecast_2': float(state[9]),
        'transshipment_cost': float(state[18]),
        'fixed_transshipment_cost': float(state[19]),
    }
    site_j = {
        'site_index': int(state[1]),
        'inventory_level': float(state[10]),
        'holding_cost': float(state[11]),
        'lost_sales_cost': float(state[12]),
        'capacity': float(state[13]),
        'fixed_order_cost': float(state[14]),
        'pipeline': float(state[15]),
        'forecast_1': float(state[16]),
        'forecast_2': float(state[17]),
        'transshipment_cost': float(state[18]),
        'fixed_transshipment_cost': float(state[19]),
    }
    return site_i, site_j


def build_global_state(replenishment_states, period, episode_length):
    site_states = [replenishment_state_to_dict(state, index)
                   for index, state in enumerate(replenishment_states)]
    inventory_values = [state['inventory_level'] for state in site_states]
    forecast_values = [state['forecast_1'] + state['forecast_2'] for state in site_states]
    total_inventory = float(sum(inventory_values))
    total_forecast = float(sum(forecast_values))
    return {
        'period': period,
        'episode_length': episode_length,
        'num_sites': len(site_states),
        'total_inventory': total_inventory,
        'total_forecast': total_forecast,
        'average_inventory': total_inventory / max(1, len(site_states)),
        'inventory_imbalance': float(max(inventory_values) - min(inventory_values)) if inventory_values else 0.0,
        'forecast_imbalance': float(max(forecast_values) - min(forecast_values)) if forecast_values else 0.0,
    }


def build_action_from_policy(state, policy, history, period, episode_length):
    replenishment_states = state[0]
    transshipment_states = state[1]
    global_state = build_global_state(replenishment_states, period, episode_length)
    actions = []
    rejected_transfers = 0

    for pair_state in transshipment_states:
        site_i, site_j = transshipment_state_to_pair(pair_state)
        raw_transfer = float(policy.transshipment_proposal(site_i, site_j, global_state))
        proposed_quantity = abs(raw_transfer)
        pair_key = f"{site_i['site_index']}-{site_j['site_index']}"
        pair_history = {
            'period': period,
            'net_pair_transfer': history['net_pair_transfer'].get(pair_key, 0.0),
            'accepted_transfer_count': history['accepted_transfer_count'],
            'rejected_transfer_count': history['rejected_transfer_count'],
        }

        if raw_transfer >= 0:
            source = site_i
            target = site_j
            direction = 1.0
        else:
            source = site_j
            target = site_i
            direction = -1.0

        accepted_quantity = float(policy.consensus_gate(source, target, proposed_quantity, pair_history))
        accepted_quantity = clamp(accepted_quantity, 0.0, proposed_quantity)
        signed_transfer = round(direction * accepted_quantity, 2)

        if proposed_quantity > 0.01 and accepted_quantity <= 0.01:
            rejected_transfers += 1
        if abs(signed_transfer) > 0.01:
            history['accepted_transfer_count'] += 1
            history['net_pair_transfer'][pair_key] = history['net_pair_transfer'].get(pair_key, 0.0) + signed_transfer

        actions.append(signed_transfer)

    for site_index, replenishment_state in enumerate(replenishment_states):
        site_state = replenishment_state_to_dict(replenishment_state, site_index)
        order_quantity = float(policy.replenishment_policy(site_state, global_state))
        order_quantity = clamp(order_quantity, 0.0, site_state['capacity'] * 3)
        actions.append(round(order_quantity, 2))

    history['rejected_transfer_count'] += rejected_transfers
    return actions, rejected_transfers


def evaluate_policy(policy, seed, parameters):
    env = InvOptEnv(seed, parameters)
    state = env.reset()
    current_ep_reward = 0.0
    current_ep_all_cost = np.array([0.0, 0.0, 0.0])
    stockout_proxy_count = 0
    history = {
        'net_pair_transfer': {},
        'accepted_transfer_count': 0,
        'rejected_transfer_count': 0,
    }

    for _ in range(1, env.epi_len + 1):
        actions, _ = build_action_from_policy(state, policy, history, env.current_period, env.epi_len)
        state, reward, done, all_cost = env.step_value(actions)
        current_ep_reward += reward
        current_ep_all_cost += np.array(all_cost)
        stockout_proxy_count += sum(1 for retailer in env.retailers if retailer.inv_level <= 0)
        if done:
            break

    avg_costs = current_ep_all_cost / env.epi_len
    fairness_debt = float(sum(abs(value) for value in history['net_pair_transfer'].values()))
    return {
        'fitness': float(-current_ep_reward / env.epi_len),
        'transshipment_cost': float(avg_costs[0]),
        'holding_lost_sales_cost': float(avg_costs[1]),
        'order_cost': float(avg_costs[2]),
        'stockout_proxy_count': int(stockout_proxy_count),
        'accepted_transfer_count': int(history['accepted_transfer_count']),
        'rejected_transfer_count': int(history['rejected_transfer_count']),
        'fairness_debt': fairness_debt,
    }
