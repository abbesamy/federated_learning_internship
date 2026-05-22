from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "experiments_output" / "experiments_results.csv"
OUTPUT_DIR = SCRIPT_DIR / "simple_plots"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_by_aggregator(rows: list[dict[str, str]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["aggregator"]].append(row)
    return grouped


def scenario_label(row: dict[str, str]) -> str:
    attack = row["attack"]
    nb_byz = row["nb_byz"]
    if attack == "clean":
        return f"clean / {nb_byz} byz"
    return f"{attack} / {nb_byz} byz"


def to_sorted_series(rows: list[dict[str, str]], metric: str):
    points = sorted(rows, key=lambda row: int(row["round"]))
    rounds = [int(row["round"]) for row in points]
    values = [float(row[metric]) for row in points]
    return rounds, values


def plot_metric_for_aggregator(aggregator: str, rows: list[dict[str, str]], metric: str, ylabel: str) -> None:
    series_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        series_by_label[scenario_label(row)].append(row)

    plt.figure(figsize=(10, 6))
    for label, series_rows in sorted(series_by_label.items()):
        rounds, values = to_sorted_series(series_rows, metric)
        plt.plot(rounds, values, marker="o", linewidth=2, label=label)

    plt.title(f"{aggregator} - {ylabel}")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{aggregator.lower()}_{metric}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    rows = read_rows(CSV_PATH)
    grouped = group_by_aggregator(rows)

    for aggregator, agg_rows in grouped.items():
        plot_metric_for_aggregator(aggregator, agg_rows, "accuracy", "Accuracy")
        plot_metric_for_aggregator(aggregator, agg_rows, "error", "Error")

    print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
