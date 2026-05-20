import json
import re
from pathlib import Path

from LLM_policy_evolution import evaluator
from LLM_policy_evolution.safe_code import PolicyValidationError


OUTPUT_ROOT = Path('Research') / 'llm_policy_runs'
POLICY_API = """def replenishment_policy(site_state, global_state):
    return order_quantity

def transshipment_proposal(source_state, target_state, global_state):
    return proposed_quantity

def consensus_gate(source_state, target_state, proposed_quantity, history):
    return accepted_quantity
"""

STATE_KEYS = """site_state keys:
- site_index, inventory_level, holding_cost, lost_sales_cost, capacity
- production_capacity, fixed_order_cost, per_unit_order_cost, pipeline
- forecast_1, forecast_2, transshipment_cost, fixed_transshipment_cost

global_state keys:
- period, episode_length, num_sites, total_inventory, total_forecast
- average_inventory, inventory_imbalance, forecast_imbalance

history keys:
- period, net_pair_transfer, accepted_transfer_count, rejected_transfer_count
"""


def scenario_dir(dataset_name):
    return OUTPUT_ROOT / f'scenario_{dataset_name}'


def manual_results_path(dataset_name, seed):
    return scenario_dir(dataset_name) / f'seed_{seed}_manual_candidates.json'


def ensure_manual_results(dataset_name, seed):
    path = manual_results_path(dataset_name, seed)
    if not path.exists():
        evaluator.main(dataset_name, seed)
    return path


def load_results(path):
    with open(path, 'r', encoding='utf-8') as result_file:
        results = json.load(result_file)
    return sorted(results, key=lambda item: item['fitness'])


def load_archive_results(dataset_name, seed):
    ensure_manual_results(dataset_name, seed)
    output_dir = scenario_dir(dataset_name)
    result_paths = [manual_results_path(dataset_name, seed)]
    result_paths.extend(sorted(output_dir.glob(f'seed_{seed}_round_*_generated_candidates.json')))

    archive = []
    seen_names = set()
    for result_path in result_paths:
        for result in load_results(result_path):
            policy_key = (result['policy_name'], result.get('policy_path', ''))
            if policy_key in seen_names:
                continue
            seen_names.add(policy_key)
            archive.append(result)
    return sorted(archive, key=lambda item: item['fitness'])


def latest_round_index(output_dir):
    round_numbers = []
    for path in output_dir.glob('round_*'):
        if path.is_dir():
            try:
                round_numbers.append(int(path.name.split('_')[1]))
            except (IndexError, ValueError):
                continue
    return max(round_numbers) if round_numbers else 0


def compact_result_table(results):
    lines = [
        '| policy | fitness | trans | hold_lost | order | stockout_proxy | accepted | rejected | fairness |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for result in results:
        lines.append(
            f"| {result['policy_name']} | {result['fitness']:.4f} | "
            f"{result['transshipment_cost']:.4f} | {result['holding_lost_sales_cost']:.4f} | "
            f"{result['order_cost']:.4f} | {result['stockout_proxy_count']} | "
            f"{result['accepted_transfer_count']} | {result['rejected_transfer_count']} | "
            f"{result['fairness_debt']:.2f} |"
        )
    return '\n'.join(lines)


def build_prompt(dataset_name, seed, round_index, top_results):
    prompt_lines = [
        f"# ConsensusCode-Evolve Mutation Prompt: round {round_index}",
        '',
        'You are improving Python inventory-control policies for a multi-site replenishment/transshipment simulator.',
        'Generate exactly 3 new candidate policy files. Keep each policy short and deterministic.',
        '',
        '## Objective',
        '- Minimize fitness: average total cost = transshipment + holding/lost-sales + order cost.',
        '- Improve consensus-seeking: avoid draining a donor site and avoid unfair repeated donor behavior.',
        '- Prefer robust service behavior without excessive orders.',
        '',
        '## Allowed API',
        '```python',
        POLICY_API.strip(),
        '```',
        '',
        '## Available State Keys',
        STATE_KEYS.strip(),
        '',
        '## Safety Rules',
        '- No imports.',
        '- No file, network, random, object attribute, class, list, or dict construction.',
        '- Only arithmetic, if statements, comparisons, and calls to abs, min, max, round.',
        '- Return nonnegative quantities from replenishment_policy and consensus_gate.',
        '- transshipment_proposal may return positive or negative signed quantities.',
        '',
        f'## Scenario And Seed',
        f'- dataset: {dataset_name}',
        f'- seed: {seed}',
        '',
        '## Current Top Policy Scores',
        compact_result_table(top_results),
        '',
        '## Current Top Policy Sources',
    ]

    for result in top_results:
        prompt_lines.extend([
            f"### {result['policy_name']}",
            '```python',
            result['source'].strip(),
            '```',
            '',
        ])

    prompt_lines.extend([
        '## Required Output Format',
        'Return exactly three fenced Python blocks. Start each block with a filename comment:',
        '',
        '```python',
        '# filename: candidate_1.py',
        'def replenishment_policy(site_state, global_state):',
        '    ...',
        '```',
        '',
        'Suggested mutation themes:',
        '1. conservative donor protection with strong service recovery,',
        '2. fairness-aware consensus with moderate transfer frequency,',
        '3. lean ordering with emergency-only transshipment.',
        '',
        'After receiving the LLM response, save the full response as `llm_response.md` in this round folder, or save each candidate `.py` file directly into `generated_candidates/`.',
    ])
    return '\n'.join(prompt_lines)


def write_round_prompt(dataset_name, seed, round_index, top_k=3):
    top_results = load_archive_results(dataset_name, seed)[:top_k]
    output_dir = scenario_dir(dataset_name) / f'round_{round_index}'
    candidates_dir = output_dir / 'generated_candidates'
    candidates_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / 'llm_mutation_prompt.md'
    instructions_path = candidates_dir / 'README.md'

    prompt = build_prompt(dataset_name, seed, round_index, top_results)
    with open(prompt_path, 'w', encoding='utf-8') as prompt_file:
        prompt_file.write(prompt)
    with open(instructions_path, 'w', encoding='utf-8') as instructions_file:
        instructions_file.write(
            'Place LLM-generated candidate .py files in this folder, then run:\n\n'
            f'python main.py {dataset_name} {seed} LLM_policy_evolution_next\n'
            '\nAlternatively, paste the full LLM answer into ../llm_response.md and run the same command.\n'
        )
    return prompt_path, candidates_dir


def extract_candidates_from_response(response_path, candidates_dir):
    with open(response_path, 'r', encoding='utf-8') as response_file:
        response = response_file.read()

    extracted_paths = []
    blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, flags=re.DOTALL | re.IGNORECASE)
    for index, block in enumerate(blocks, start=1):
        source = block.strip()
        if not source:
            continue
        filename = f'candidate_{index}.py'
        lines = source.splitlines()
        filename_match = re.match(r'#\s*filename:\s*([A-Za-z0-9_\-.]+\.py)', lines[0])
        if filename_match:
            filename = filename_match.group(1)
            source = '\n'.join(lines[1:]).strip()
        if not source:
            continue
        candidate_path = candidates_dir / filename
        with open(candidate_path, 'w', encoding='utf-8') as candidate_file:
            candidate_file.write(source + '\n')
        extracted_paths.append(candidate_path)
    return extracted_paths


def evaluate_generated_candidates(dataset_name, seed, round_index, candidates_dir):
    results = []
    for policy_path in sorted(candidates_dir.glob('*.py')):
        try:
            result = evaluator.evaluate_policy_file(policy_path, dataset_name, seed)
        except PolicyValidationError as error:
            result = {
                'policy_name': policy_path.stem,
                'policy_path': str(policy_path),
                'error': str(error),
            }
        results.append(result)

    valid_results = [result for result in results if 'error' not in result]
    valid_results.sort(key=lambda item: item['fitness'])
    rejected = [result for result in results if 'error' in result]
    output_dir = scenario_dir(dataset_name) / f'round_{round_index}'
    output_dir.mkdir(parents=True, exist_ok=True)

    if valid_results:
        json_path, csv_path = evaluator.write_results(
            valid_results,
            dataset_name,
            seed,
            output_root=OUTPUT_ROOT,
            run_label=f'round_{round_index}_generated_candidates',
        )
    else:
        json_path = None
        csv_path = None

    if rejected:
        rejected_path = output_dir / 'rejected_candidates.json'
        with open(rejected_path, 'w', encoding='utf-8') as rejected_file:
            json.dump(rejected, rejected_file, indent=2)

    return valid_results, rejected, json_path, csv_path


def main(dataset_name, seed):
    output_dir = scenario_dir(dataset_name)
    latest_round = latest_round_index(output_dir)

    if latest_round > 0:
        latest_candidates_dir = output_dir / f'round_{latest_round}' / 'generated_candidates'
        latest_result_path = output_dir / f'seed_{seed}_round_{latest_round}_generated_candidates.json'
        response_path = output_dir / f'round_{latest_round}' / 'llm_response.md'
        if response_path.exists() and not [path for path in latest_candidates_dir.glob('*.py')]:
            extracted_paths = extract_candidates_from_response(response_path, latest_candidates_dir)
            print(f'Extracted {len(extracted_paths)} candidate files from {response_path}')
        candidate_files = [path for path in latest_candidates_dir.glob('*.py')]
        if candidate_files and not latest_result_path.exists():
            print('----------LLM policy evolution evaluate generated candidates----------')
            print(f'Round: {latest_round}')
            valid_results, rejected, json_path, csv_path = evaluate_generated_candidates(
                dataset_name, seed, latest_round, latest_candidates_dir)
            for result in valid_results:
                print(f"{result['policy_name']}: fitness={result['fitness']:.4f}, "
                      f"costs=({result['transshipment_cost']:.4f}, "
                      f"{result['holding_lost_sales_cost']:.4f}, {result['order_cost']:.4f})")
            print(f'Valid candidates: {len(valid_results)}')
            print(f'Rejected candidates: {len(rejected)}')
            if json_path is not None:
                print(f'Saved JSON: {json_path}')
                print(f'Saved CSV: {csv_path}')
            return
        if not latest_result_path.exists():
            print('----------LLM policy evolution waiting for generated candidates----------')
            print(f'Round: {latest_round}')
            print(f'Prompt: {output_dir / f"round_{latest_round}" / "llm_mutation_prompt.md"}')
            print(f'Candidate folder: {latest_candidates_dir}')
            print('No generated .py candidates found yet.')
            print('Save candidate .py files into the candidate folder, or paste the LLM answer into llm_response.md in the round folder, then run this command again.')
            return

    next_round = latest_round + 1
    prompt_path, candidates_dir = write_round_prompt(dataset_name, seed, next_round)
    candidate_files = [path for path in candidates_dir.glob('*.py')]

    print('----------LLM policy evolution next round----------')
    print(f'Prompt saved: {prompt_path}')
    print(f'Candidate folder: {candidates_dir}')

    if not candidate_files:
        print('No generated .py candidates found yet.')
        print('Open the prompt, ask an LLM for candidates, save them into the candidate folder, then run this command again.')
        return

    valid_results, rejected, json_path, csv_path = evaluate_generated_candidates(
        dataset_name, seed, next_round, candidates_dir)
    for result in valid_results:
        print(f"{result['policy_name']}: fitness={result['fitness']:.4f}, "
              f"costs=({result['transshipment_cost']:.4f}, "
              f"{result['holding_lost_sales_cost']:.4f}, {result['order_cost']:.4f})")
    print(f'Valid candidates: {len(valid_results)}')
    print(f'Rejected candidates: {len(rejected)}')
    if json_path is not None:
        print(f'Saved JSON: {json_path}')
        print(f'Saved CSV: {csv_path}')


if __name__ == '__main__':
    main('sN2h_1_5b2', 0)