import csv
import json
from pathlib import Path

from LLM_policy_evolution.policy_adapter import evaluate_policy
from LLM_policy_evolution.safe_code import PolicyValidationError, load_policy_file
from Utils.ScenarioDesign_replenish_transship import ScenarioDesign_replenish_transship


DEFAULT_CANDIDATE_DIR = Path(__file__).resolve().parent / 'seed_policies'
DEFAULT_OUTPUT_ROOT = Path('Research') / 'llm_policy_runs'


def evaluate_policy_file(policy_path, dataset_name, seed):
    policy, source = load_policy_file(policy_path)
    parameters = ScenarioDesign_replenish_transship(dataset_name).get_parameter(seed)
    metrics = evaluate_policy(policy, seed, parameters)
    metrics['policy_name'] = policy_path.stem
    metrics['policy_path'] = str(policy_path)
    metrics['source_lines'] = len(source.splitlines())
    metrics['source'] = source
    return metrics


def write_results(results, dataset_name, seed, output_root=DEFAULT_OUTPUT_ROOT, run_label='manual_candidates'):
    output_dir = output_root / f'scenario_{dataset_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'seed_{seed}_{run_label}.json'
    csv_path = output_dir / f'seed_{seed}_{run_label}.csv'
    source_dir = output_dir / f'seed_{seed}_{run_label}_sources'
    source_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=2)

    for result in results:
        source_path = source_dir / f"{result['policy_name']}.py"
        with open(source_path, 'w', encoding='utf-8') as source_file:
            source_file.write(result['source'])

    fieldnames = [
        'policy_name', 'fitness', 'transshipment_cost', 'holding_lost_sales_cost', 'order_cost',
        'stockout_proxy_count', 'accepted_transfer_count', 'rejected_transfer_count',
        'fairness_debt', 'source_lines', 'policy_path'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    return json_path, csv_path


def main(dataset_name, seed, candidate_dir=DEFAULT_CANDIDATE_DIR):
    candidate_dir = Path(candidate_dir)
    results = []
    for policy_path in sorted(candidate_dir.glob('*.py')):
        if policy_path.name == '__init__.py':
            continue
        try:
            result = evaluate_policy_file(policy_path, dataset_name, seed)
        except PolicyValidationError as error:
            result = {
                'policy_name': policy_path.stem,
                'policy_path': str(policy_path),
                'error': str(error),
            }
        results.append(result)

    valid_results = [result for result in results if 'error' not in result]
    valid_results.sort(key=lambda item: item['fitness'])
    json_path, csv_path = write_results(valid_results, dataset_name, seed)

    print('----------LLM policy evolution seed evaluation----------')
    print(f'Dataset: {dataset_name}, seed: {seed}')
    for result in valid_results:
        print(f"{result['policy_name']}: fitness={result['fitness']:.4f}, "
              f"costs=({result['transshipment_cost']:.4f}, "
              f"{result['holding_lost_sales_cost']:.4f}, {result['order_cost']:.4f}), "
              f"fairness_debt={result['fairness_debt']:.2f}")
    print(f'Saved JSON: {json_path}')
    print(f'Saved CSV: {csv_path}')
    if len(valid_results) != len(results):
        print(f'Rejected {len(results) - len(valid_results)} invalid policy files.')


if __name__ == '__main__':
    main('sN2h_1_5b2', 0)
