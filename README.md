# InventoryControl

InventoryControl is a research codebase for multi-location inventory control. It combines simulation-based evaluation with genetic programming, deep reinforcement learning, particle swarm optimization, and baseline inventory policies to study replenishment, transshipment, rental, RFQ, and pricing decisions.

The project is organized around repeatable experiments: choose a scenario, choose a random seed, choose an algorithm, and run the dispatcher in `main.py`.

## What This Project Does

The repository models inventory systems with multiple retailers or warehouses, stochastic demand, lead times, holding costs, lost-sales costs, ordering costs, and transshipment costs. Candidate policies are trained or tested inside inventory simulators, then saved for later analysis.

The main research question is how different policy-search methods perform on inventory-control decisions such as:

- when and how much to replenish;
- whether inventory should be transshipped between locations;
- how rental, RFQ, and pricing decisions affect cost;
- how learned rules generalize from training scenarios to test scenarios.

## Main Methods

| Method family | Main folders | Description |
| --- | --- | --- |
| CCGP | `CCGP_niching/`, `CCGP_niching_rental/`, `CCGP_niching_rental_original/` | Cooperative coevolutionary genetic programming variants for replenishment and transshipment policies. |
| MTGP | `MTGP_niching/`, `MTGP_niching_rental/`, `MTGP_niching_rental_original/` | Multi-tree genetic programming variants, usually evolving separate rule trees for different inventory decisions. |
| InventoryGP | `MTGP_niching_replenish_transship_price/` | Extended GP method that combines replenishment, transshipment, and pricing-related decisions. |
| RFQ and pricing GP | `MTGP_niching_rental_RFQ/`, `MTGP_niching_rental_RFQ_price/` | Rental and request-for-quote variants with additional pricing logic. |
| DRL | `DRL/` | A PPO-based reinforcement learning baseline for inventory-control actions. |
| PSO | `PSO_rental/` | Particle swarm optimization baseline for rental scenarios. |
| sS policy | `sSPolicy/` | Threshold-style inventory policy baseline. |

Most algorithm folders contain their own training driver, simulator, policy operators, selection logic, and result-saving utilities. This duplication is intentional in a research workflow because each variant may need different terminals, actions, fitness calculations, or saved outputs.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Main experiment dispatcher. It routes `<dataset_name> <seed> <algo>` to the selected method. |
| `Utils/ScenarioDesign*.py` | Scenario definitions and parameter generation for synthetic and real-data experiments. |
| `Utils/teckwah.csv` | Real demand data used by the Teckwah scenarios. |
| `resultsAnalysis/` | Plotting, convergence analysis, statistical tests, rule-size analysis, and visualization scripts. |
| `wilcoxonTest/` | Wilcoxon significance testing utilities. |
| `Sanwa/`, `Taiyo/` | Company-specific inventory problem assets or experiments. |
| `test policy *.py` | Standalone scripts for testing random or GP-based policies. |
| `S2Demo.py` | Demonstration script for a selected scenario. |

## Running Experiments

Run commands from the repository root.

```powershell
python main.py <dataset_name> <seed> <algo>
```

Examples:

```powershell
python main.py teckwah_training 0 MTGP_niching
python main.py teckwah_training 0 CCGP_niching
python main.py sN2h_1_5b2 0 InventoryGP
python main.py teckwah_training 0 DRL
python main.py teckwah_training 0 PSO_rental
python main.py teckwah_test 0 InventoryGP-test
```

Common algorithm names accepted by `main.py` include:

- `CCGP_niching`
- `CCGP_niching_rental`
- `MTGP_niching`
- `MTGP_niching_rental`
- `MTGP_niching_rental_RFQ`
- `redGP`
- `InventoryGP`
- `InventoryGP-test`
- `redGP-test`
- `PSO_rental`
- `DRL`
- `testRuleMTGP`
- `testRuleCCGP`
- `testRuleDRL`
- `trainRulesSPolicy`
- `S2Demo`

Some dispatcher options run multiple seeds internally, such as `PSO_rental_3`, `MTGP_niching_rental_2`, and original-method comparison modes.

## Scenarios

The base scenario format is defined in `Utils/ScenarioDesign.py`.

Real-data scenarios:

- `teckwah_training`
- `teckwah_test`

Synthetic scenario names encode demand scale, number of retailers, holding costs, and lost-sales multiplier. For example:

```text
sN2h_1_5b2
```

This represents a small-demand, two-retailer scenario with holding costs `1` and `5`, and a lost-sales multiplier of `2`. Larger examples include names such as `mN3h_10_50_50b3` or `lN2h_1_5b2`.

Several method extensions use their own scenario-design files, such as rental, RFQ, RFQ price, and replenish-transship-price variants.

## Method Workflow

1. `main.py` reads the dataset name, seed, and algorithm name.
2. The selected method loads scenario parameters from `Utils/ScenarioDesign*.py`.
3. The method builds candidate policy representations or action maps.
4. An inventory simulator evaluates policy performance over one or more episodes.
5. Training saves best individuals, policies, fitness histories, running time, or action maps under method-specific result folders.
6. Analysis scripts in `resultsAnalysis/` generate plots, compare test performance, and run statistical tests.

For GP methods, policy quality is usually measured as a cost-minimization fitness value. For DRL, rewards are shaped around the same inventory-cost objective and PPO updates the policy from simulated episodes.

## Dependencies

The repository does not currently include a `requirements.txt`. Based on imports, the main dependencies include:

- Python 3.x
- `numpy`
- `pandas`
- `deap`
- `torch`
- `matplotlib`
- `scipy`
- `pytz`

Install missing packages in your preferred environment, for example:

```powershell
pip install numpy pandas deap torch matplotlib scipy pytz
```

## Development Notes

- Keep runs reproducible by passing explicit seeds and preserving seed setup inside each method.
- Start with small synthetic scenarios when debugging because full training runs can be expensive.
- Many algorithm folders share similar file names but differ in important details. Compare variants before moving logic across folders.
- On Windows, keep multiprocessing entry points guarded and preserve `freeze_support()` behavior in runnable scripts.

## Simulator Architecture

The simulator code is being moved toward a shared-core structure. Common inventory mechanics live in `Utils/inventory_core.py`, including demand models, Teckwah demand handling, basic retailer state transitions, and rental-aware retailer state transitions.

Method folders should keep their method-specific pieces, such as GP tree evaluation, PPO action maps, PSO action vectors, state feature construction, result saving, and experiment parameters. Shared physical mechanics such as demand generation, pipeline arrivals, capacity limits, rental shortage support, and demand-record selection should be reused from the core module so method comparisons are not affected by copied simulator bugs.

Current first-stage migrations use the shared core in the base CCGP, MTGP, sS, DRL, and main rental CCGP/MTGP/PSO simulators. RFQ, pricing, and niching variants still have method-specific simulator logic and should be migrated gradually with fixed-action regression checks.

## Copilot Agent

This repository includes a workspace custom agent at `.github/agents/inventory-control-coder.agent.md`. Use it in VS Code when working on coding, debugging, documentation, or method-improvement tasks for this project.