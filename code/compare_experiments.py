from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = PROJECT_ROOT / "code" / "experiment_outputs" / "combined_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "code" / "experiment_outputs" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all experiment curves from the combined results CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help="Path to the combined results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where comparison plots will be written.",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "error"],
        default="accuracy",
        help="Primary metric to compare.",
    )
    return parser.parse_args()


def build_label(row: pd.Series) -> str:
    return f"{row['scenario']} | {row['aggregator']} | {row['attack']}"


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    for scenario_name, group in df.groupby("scenario", sort=False):
        ordered = group.sort_values("round")
        first_row = ordered.iloc[0]
        label = build_label(first_row)

        line_style = "--" if int(first_row["nb_byz"]) == 0 else "-"
        linewidth = 2.8 if int(first_row["nb_byz"]) == 0 else 2.2
        ax.plot(
            ordered["round"],
            ordered[metric],
            label=label,
            linewidth=linewidth,
            linestyle=line_style,
            alpha=0.95,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Comparison of all scenarios - {metric.capitalize()}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_final_round(df: pd.DataFrame) -> pd.DataFrame:
    final_rounds = df.sort_values("round").groupby("scenario", as_index=False).tail(1)
    summary = final_rounds[
        [
            "scenario",
            "dataset",
            "distribution",
            "aggregator",
            "attack",
            "nb_honest",
            "nb_byz",
            "round",
            "accuracy",
            "error",
        ]
    ].copy()
    summary = summary.sort_values("accuracy", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank_by_accuracy", range(1, len(summary) + 1))
    return summary


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required_columns = {"scenario", "round", "accuracy", "error", "aggregator", "attack", "nb_byz"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"CSV is missing columns: {', '.join(sorted(missing_columns))}")

    accuracy_plot = output_dir / "all_scenarios_accuracy.png"
    error_plot = output_dir / "all_scenarios_error.png"
    summary_csv = output_dir / "final_round_summary.csv"

    plot_metric(df, args.metric, accuracy_plot if args.metric == "accuracy" else error_plot)
    plot_metric(df, "error", error_plot)
    plot_metric(df, "accuracy", accuracy_plot)

    summary = summarize_final_round(df)
    summary.to_csv(summary_csv, index=False)

    print(f"Comparison plots saved to: {output_dir}")
    print(f"Accuracy plot: {accuracy_plot}")
    print(f"Error plot: {error_plot}")
    print(f"Summary CSV: {summary_csv}")
    print("Top scenarios by final accuracy:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
