"""Run from repository root; no changes to root main.py or multiprocessing behavior."""

import argparse
import json
import multiprocessing
from pathlib import Path

from .policy import load_policy
from .scenario import Settings
from .training import (EXPERIMENT_SCHEMA, SERVICE_CONSTRAINT, Search,
                       jsonable, run_evaluation, run_training)


def seed_list(value):
    try:
        result = tuple(int(item) for item in value.split(","))
        if not result or any(seed < 0 or seed >= 2**32 - 1000 for seed in result):
            raise ValueError()
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative integer seeds") from exc


def topology(value):
    if value == "complete":
        return None
    if value == "none":
        return ()
    try:
        return tuple(tuple(int(site) for site in edge.split("-")) for edge in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use complete, none, or 0-1,1-2,...") from exc


def parser():
    defaults = Search()
    result = argparse.ArgumentParser(
        description="PACT-GP: three-tree genetic programming for peer-agent coordination")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("train", "smoke", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path,
                             help="Must be empty; defaults to package outputs/<command>")
        command.add_argument("--scenario", default=None)
        command.add_argument("--rounds", type=int, default=None)
        command.add_argument("--periods", type=int, default=None,
                             help="Execute a prefix; keep full benchmark generation streams")
        command.add_argument("--topology", type=topology, default=argparse.SUPPRESS)
        command.add_argument("--allow-oracle-forecasts", action="store_true", default=None)
        command.add_argument("--service-target", type=float, default=None)
        command.add_argument("--no-trace", action="store_true")
        if name == "evaluate":
            command.add_argument("--policy", type=Path, required=True)
            command.add_argument("--config", type=Path,
                                 help="Replay saved scenario/settings; explicit CLI overrides allowed")
            command.add_argument("--seeds", type=seed_list, default=(2001, 2002))
        else:
            command.add_argument("--seed", type=int, default=0, help="DEAP search seed")
            command.add_argument("--population", type=int,
                                 default=6 if name == "smoke" else defaults.population)
            command.add_argument("--generations", type=int,
                                 default=1 if name == "smoke" else defaults.generations)
            command.add_argument("--train-seeds", type=seed_list, default=(11, 12))
            command.add_argument("--validation-seeds", type=seed_list, default=(1001, 1002))
            command.add_argument("--test-seeds", type=seed_list, default=(2001, 2002))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    saved = {}
    if args.command == "evaluate" and args.config:
        saved = json.loads(args.config.read_text(encoding="utf-8"))
        if (saved.get("schema") != EXPERIMENT_SCHEMA
                or saved.get("service_constraint") != SERVICE_CONSTRAINT):
            raise ValueError(
                "Expected peer-experiment-v2 with site-level service constraint; "
                "old network-service experiment configs are not replay-compatible. "
                "Re-score an old policy with explicit current settings instead.")
    settings_dict = saved.get("settings", {})
    for name in ("rounds", "periods", "service_target", "allow_oracle_forecasts"):
        value = getattr(args, name)
        if value is not None:
            settings_dict[name] = value
    if hasattr(args, "topology"):
        settings_dict["topology"] = args.topology
    settings = Settings(**settings_dict)
    scenario = args.scenario or saved.get("scenario", "sN2h_1_5b2")
    output = args.output_dir or Path(__file__).parent / "outputs" / args.command
    if args.command == "evaluate":
        summary = run_evaluation(load_policy(args.policy), scenario, args.seeds,
                                 settings, output, trace=not args.no_trace)
        print(json.dumps(jsonable({"output": str(output.resolve()), "summary": summary}),
                         indent=2, allow_nan=False))
    else:
        chosen, metrics = run_training(
            scenario, settings, Search(seed=args.seed, population=args.population,
                                      generations=args.generations),
            args.train_seeds, args.validation_seeds, args.test_seeds, output,
            trace=not args.no_trace)
        print(json.dumps({"output": str(output.resolve()), "trees": chosen.expressions(),
                          "splits": {name: {key: summary[key] for key in
                                           ("worst_service_violation", "mean_period_cost",
                                            "mean_network_fill_rate", "mean_site_fill_rates", "size")}
                                     for name, summary in metrics.items()}}, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
