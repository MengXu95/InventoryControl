"""Small, explicit serial GP search with fixed common random numbers and splits."""

import copy
import importlib.metadata
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .environment import SERVICE_CONSTRAINT, Environment
from .policy import (canonical_json, crossover, mutate, policy_document,
                     random_individual, save_policy, seeded_random, template)
from .scenario import Settings, parameters_for

EXPERIMENT_SCHEMA = "peer-experiment-v2"


@dataclass(frozen=True)
class Search:
    seed: int = 0
    population: int = 20
    generations: int = 5
    crossover_probability: float = 0.8
    mutation_probability: float = 0.15
    tournament: int = 3

    def __post_init__(self):
        if type(self.population) is not int or self.population < 2:
            raise ValueError("population must be >= 2")
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("generations must be nonnegative")
        if type(self.tournament) is not int or self.tournament < 1:
            raise ValueError("tournament must be positive")
        if not (0 <= self.crossover_probability <= 1
                and 0 <= self.mutation_probability <= 1
                and self.crossover_probability + self.mutation_probability <= 1):
            raise ValueError("Variation probabilities must sum to <= 1")


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(canonical_json(jsonable(value)), encoding="utf-8")


def prepare_output(path):
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty (prevent accidental overwrite): {output}")
    return output


def validate_splits(train, validation, test):
    splits = (tuple(train), tuple(validation), tuple(test))
    for seeds in splits:
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Each split needs nonempty distinct seeds")
        if any(type(seed) is not int or not 0 <= seed < 2**32 - 1000 for seed in seeds):
            raise ValueError("Seeds must be integers in [0, 2**32-1000)")
    if any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation and held-out test seeds must be disjoint")
    return splits


def evaluate_batch(individual, scenario, seeds, settings, trace=False):
    episodes, traces = [], []
    for seed in seeds:
        environment = Environment(parameters_for(scenario, seed), seed, settings)
        metrics, events = environment.run(individual, trace=trace)
        episodes.append(metrics)
        if trace:
            traces.append({"seed": seed, "events": events})
    summary = {
        "service_constraint": SERVICE_CONSTRAINT,
        "worst_service_violation": max(m["service_violation"] for m in episodes),
        "mean_period_cost": sum(m["mean_period_cost"] for m in episodes) / len(episodes),
        "mean_network_fill_rate": sum(m["network_fill_rate"] for m in episodes) / len(episodes),
        "mean_site_fill_rates": [
            sum(m["site_metrics"][site]["fill_rate"] for m in episodes) / len(episodes)
            for site in range(len(episodes[0]["site_metrics"]))
        ],
        "size": individual.size,
        "episodes": episodes,
    }
    return summary, traces


def fitness(summary):
    # Lexicographic minimization, NOT a weighted service/cost compromise.
    return (summary["worst_service_violation"], summary["mean_period_cost"], summary["size"])


def versions():
    return {"python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "deap": importlib.metadata.version("deap"),
            "pandas": importlib.metadata.version("pandas")}


def run_training(scenario, settings, search, train_seeds, validation_seeds, test_seeds,
                 output_dir, trace=True):
    train_seeds, validation_seeds, test_seeds = validate_splits(
        train_seeds, validation_seeds, test_seeds)
    output = prepare_output(output_dir)
    config = {
        "schema": EXPERIMENT_SCHEMA, "mode": "train", "scenario": scenario,
        "service_constraint": SERVICE_CONSTRAINT,
        "settings": settings.document(), "search": asdict(search),
        "splits": {"train": train_seeds, "validation": validation_seeds, "test": test_seeds},
        "versions": versions(), "seed_rotation": False,
        "fitness": ["worst_service_violation", "mean_period_cost", "total_nodes"],
        "selection": "validation lexicographic best among unique generation champions",
        "scenario_parameters": {str(seed): parameters_for(scenario, seed)
                                for seed in train_seeds + validation_seeds + test_seeds},
    }
    write_json(output / "config.json", config)
    cache, individuals, champions, history = {}, {}, {}, []

    def key(individual):
        return tuple(str(tree) for tree in individual)

    def score(individual):
        identity = key(individual)
        if identity not in cache:
            cache[identity], _ = evaluate_batch(individual, scenario, train_seeds, settings)
            individuals[identity] = copy.deepcopy(individual)
        return fitness(cache[identity])

    with seeded_random(search.seed):
        population = [template()] + [random_individual() for _ in range(search.population - 1)]
        for generation in range(search.generations + 1):
            population.sort(key=lambda individual: (score(individual), key(individual)))
            champion = copy.deepcopy(population[0])
            champions[key(champion)] = champion
            history.append({"generation": generation, "best": cache[key(champion)],
                            "trees": champion.expressions(),
                            "unique_evaluations_so_far": len(cache)})
            save_policy(champion, output / f"generation_{generation:03d}_policy.json")
            if generation == search.generations:
                break

            def select():
                contestants = random.choices(population, k=search.tournament)
                return min(contestants, key=lambda ind: (score(ind), key(ind)))

            offspring = [champion]  # one elite, same FIXED evaluation batch
            while len(offspring) < search.population:
                draw = random.random()
                if draw < search.crossover_probability:
                    children = crossover(select(), select())
                elif draw < search.crossover_probability + search.mutation_probability:
                    children = (mutate(select()),)
                else:
                    children = (copy.deepcopy(select()),)
                offspring.extend(children[:search.population - len(offspring)])
            population = offspring
    # No cross-batch HallOfFame; all gen champions are compared on ONE validation batch.
    validation = []
    for identity, individual in sorted(champions.items()):
        summary, _ = evaluate_batch(individual, scenario, validation_seeds, settings)
        validation.append({"individual": individual, "summary": summary, "key": identity})
    chosen = min(validation, key=lambda row: (fitness(row["summary"]), row["key"]))["individual"]
    save_policy(chosen, output / "best_policy.json")
    write_json(output / "history.json", history)
    write_json(output / "training_evaluations.json", [
        {"policy": policy_document(individuals[identity]), "metrics": cache[identity]}
        for identity in sorted(cache)])
    write_json(output / "validation_candidates.json", [
        {"trees": row["individual"].expressions(), "metrics": row["summary"]}
        for row in validation])
    results = {}
    for split, seeds in (("train", train_seeds), ("validation", validation_seeds), ("test", test_seeds)):
        summary, traces = evaluate_batch(chosen, scenario, seeds, settings, trace=trace)
        results[split] = summary
        if trace:
            write_json(output / f"traces_{split}.json", traces)
    write_json(output / "metrics.json", results)
    return chosen, results


def run_evaluation(individual, scenario, seeds, settings, output_dir, trace=True):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and distinct")
    output = prepare_output(output_dir)
    write_json(output / "config.json", {
        "schema": EXPERIMENT_SCHEMA, "mode": "evaluate", "scenario": scenario,
        "service_constraint": SERVICE_CONSTRAINT,
        "settings": settings.document(), "seeds": seeds, "versions": versions(),
        "scenario_parameters": {str(seed): parameters_for(scenario, seed) for seed in seeds},
    })
    save_policy(individual, output / "policy.json")
    summary, traces = evaluate_batch(individual, scenario, seeds, settings, trace=trace)
    write_json(output / "metrics.json", summary)
    if trace:
        write_json(output / "traces.json", traces)
    return summary
