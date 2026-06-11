from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from moteur_experiences import (  # noqa: E402
    DEFAULT_COMMON_CONFIG,
    build_config,
    load_common_config,
    load_json,
    resolve_repo_path,
    run_scenario,
)


DEFAULT_OUTPUT_DIR = THIS_DIR / "resultats"
DEFAULT_SAMPLES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same experiment multiple times with different seeds and plot the average curve."
    )
    parser.add_argument(
        "--scenario-file",
        type=str,
        required=True,
        help="Path to the scenario JSON file to run.",
    )
    parser.add_argument(
        "--common-config",
        type=str,
        default=str(DEFAULT_COMMON_CONFIG),
        help="Path to the common JSON config.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of repetitions with different seeds.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed to use.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Difference between two consecutive seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV files and plots will be saved.",
    )
    parser.add_argument("--rounds", type=int, help="Override number of rounds.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval.")
    return parser.parse_args()


def build_run_config(common_config: dict, scenario_config: dict, seed: int) -> dict:
    config = build_config(common_config, scenario_config)
    config["seed"] = seed
    return config


def run_repeated_experiments(
    common_config: dict,
    scenario_config: dict,
    runs: int,
    seed_start: int,
    seed_step: int,
) -> pd.DataFrame:
    records: list[dict] = []
    for run_index in range(runs):
        seed = seed_start + run_index * seed_step
        config = build_run_config(common_config, scenario_config, seed)
        config["name"] = f"{scenario_config['name']}_run_{run_index + 1:02d}"

        print(f"Running {config['name']} with seed={seed}")
        run_records = run_scenario(config)
        for record in run_records:
            record["run_index"] = run_index + 1
            record["seed"] = seed
            record["base_scenario"] = scenario_config["name"]
        records.extend(run_records)

    return pd.DataFrame(records)


def aggregate_mean(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["base_scenario", "round"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        error_mean=("error", "mean"),
        error_std=("error", "std"),
        runs=("run_index", "nunique"),
    )
    grouped["accuracy_std"] = grouped["accuracy_std"].fillna(0.0)
    grouped["error_std"] = grouped["error_std"].fillna(0.0)
    return grouped.sort_values(["base_scenario", "round"]).reset_index(drop=True)


def plot_average_curve(mean_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    round_max = int(mean_df["round"].max()) if not mean_df.empty else 1
    x_max = max(round_max, 2)

    for scenario_name, group in mean_df.groupby("base_scenario", sort=False):
        ordered = group.sort_values("round")
        ax.plot(ordered["round"], ordered["accuracy_mean"], linewidth=2.5, label=f"{scenario_name} (moyenne)")
        ax.fill_between(
            ordered["round"],
            ordered["accuracy_mean"] - ordered["accuracy_std"],
            ordered["accuracy_mean"] + ordered["accuracy_std"],
            alpha=0.2,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy moyenne")
    ax.set_title("Courbe moyenne sur plusieurs seeds")
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    scenario_path = resolve_repo_path(args.scenario_file)
    common_config = load_common_config(resolve_repo_path(args.common_config))
    scenario_config = load_json(scenario_path)

    if args.rounds is not None:
        common_config["rounds"] = args.rounds
    if args.batch_size is not None:
        common_config["batch_size"] = args.batch_size
    if args.eval_interval is not None:
        common_config["eval_interval"] = args.eval_interval

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = run_repeated_experiments(
        common_config=common_config,
        scenario_config=scenario_config,
        runs=args.runs,
        seed_start=args.seed_start,
        seed_step=args.seed_step,
    )

    raw_csv = output_dir / "resultats_bruts.csv"
    mean_csv = output_dir / "courbe_moyenne.csv"
    plot_path = output_dir / "courbe_moyenne.png"

    runs_df.to_csv(raw_csv, index=False)

    mean_df = aggregate_mean(runs_df)
    mean_df.to_csv(mean_csv, index=False)
    plot_average_curve(mean_df, plot_path)

    print(f"Résultats bruts enregistrés dans : {raw_csv}")
    print(f"Courbe moyenne enregistrée dans : {mean_csv}")
    print(f"Graphique enregistré dans : {plot_path}")


if __name__ == "__main__":
    main()