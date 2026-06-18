from __future__ import annotations

import argparse
import re
from pathlib import Path

from moteur_experiences import (
    DEFAULT_COMMON_CONFIG,
    DEFAULT_SCENARIO_INDEX,
    build_config,
    load_common_config,
    load_json,
    load_scenario_index,
    plot_records,
    resolve_repo_path,
    run_scenario,
    save_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment scenarios described in JSON files.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument("--all", action="store_true", help="Run all scenarios from the index.")
    parser.add_argument("--scenario", nargs="*", help="Scenario name(s) to run.")
    parser.add_argument("--scenario-file", type=str, help="Run a scenario JSON file directly.")
    parser.add_argument("--common-config", type=str, default=str(DEFAULT_COMMON_CONFIG), help="Path to the common JSON config.")
    parser.add_argument("--scenario-index", type=str, default=str(DEFAULT_SCENARIO_INDEX), help="Path to the scenario index JSON.")
    parser.add_argument("--rounds", type=int, help="Override number of rounds.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval.")
    parser.add_argument("--output-dir", type=str, help="Override output directory.")
    return parser.parse_args()


def sanitize_name(value: object) -> str:
    value_str = str(value) if value is not None else ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value_str.strip().lower())
    return normalized.strip("_") or "resultats"


def build_output_filename(scenario_configs: list[dict[str, object]]) -> str:
    if not scenario_configs:
        return "resultats_combines.csv"

    attacks = {str(config.get("attack", "")).strip() for config in scenario_configs}
    distributions = {str(config.get("distribution", "")).strip() for config in scenario_configs}

    if len(attacks) == 1 and len(distributions) == 1:
        attack_name = sanitize_name(next(iter(attacks)))
        distribution_name = sanitize_name(next(iter(distributions)))
        return f"resultats_{attack_name}_{distribution_name}.csv"

    return "resultats_combines.csv"


def main() -> None:
    args = parse_args()

    common_config = load_common_config(resolve_repo_path(args.common_config))
    if args.output_dir:
        common_config["output_dir"] = args.output_dir
    if args.rounds is not None:
        common_config["rounds"] = args.rounds
    if args.batch_size is not None:
        common_config["batch_size"] = args.batch_size
    if args.eval_interval is not None:
        common_config["eval_interval"] = args.eval_interval

    if args.scenario_file:
        scenario_path = resolve_repo_path(args.scenario_file)
        scenario_entries = [{"name": scenario_path.stem, "file": args.scenario_file, "description": "Custom scenario file"}]
    else:
        scenario_index = load_scenario_index(resolve_repo_path(args.scenario_index))
        if args.list:
            for entry in scenario_index:
                print(f"{entry['name']}: {entry.get('description', '')}")
            return
        if args.all or not args.scenario:
            scenario_entries = scenario_index
        else:
            wanted = set(args.scenario)
            scenario_entries = [entry for entry in scenario_index if entry["name"] in wanted]
            missing = wanted - {entry["name"] for entry in scenario_entries}
            if missing:
                raise SystemExit(f"Unknown scenario(s): {', '.join(sorted(missing))}")

    all_records = []
    scenario_configs: list[dict[str, object]] = []
    for entry in scenario_entries:
        scenario_config = load_json(resolve_repo_path(entry["file"]))
        scenario_configs.append(scenario_config)
        config = build_config(common_config, scenario_config)
        print(f"Running scenario: {config['name']}")
        records = run_scenario(config)
        all_records.extend(records)

    # Modification ici : Le dossier par défaut 'resultats_experiences' est désormais 
    # au même niveau que le répertoire 'code', et non plus à l'intérieur.
    output_dir = resolve_repo_path(common_config.get("output_dir", "resultats_experiences"))

    output_filename = build_output_filename(scenario_configs)
    csv_path = save_records(all_records, output_dir, filename=output_filename)
    plot_records(all_records, output_dir)
    print(f"Results saved to: {csv_path}")
    print(f"Graphiques enregistrés dans : {output_dir / 'graphiques'}")


if __name__ == "__main__":
    main()