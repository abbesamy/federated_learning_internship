from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STRUCTURE DES DOSSIERS (Mise à jour Racine)
# ==============================================================================
# __file__ est dans "PROJECT_ROOT/code/nom_du_script.py"
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Dossier global des résultats (au même niveau que code/)
RESULTS_DIR = PROJECT_ROOT / "resultats_experiences"

INPUT_CSV = RESULTS_DIR / "resultats_combines.csv"
PLOTS_DIR = RESULTS_DIR / "graphiques"
COMPARISON_DIR = RESULTS_DIR / "comparaison"
# ==============================================================================


def plot_per_scenario(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    round_max = int(df["round"].max()) if not df.empty else 1
    x_max = max(round_max, 2)

    for scenario_name, group in df.groupby("scenario"):
        ordered = group.sort_values("round")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ordered["round"], ordered["accuracy"], color="tab:blue", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title(scenario_name)
        ax.set_xlim(1, x_max)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{scenario_name}.png", dpi=150)
        plt.close(fig)


def plot_comparison(df: pd.DataFrame) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    round_max = int(df["round"].max()) if not df.empty else 1
    x_max = max(round_max, 2)

    for scenario_name, group in df.groupby("scenario", sort=False):
        ordered = group.sort_values("round")
        first_row = ordered.iloc[0]
        label = f"{first_row['scenario']} | {first_row['aggregator']} | {first_row['attack']}"
        line_style = "--" if int(first_row["nb_byz"]) == 0 else "-"
        linewidth = 2.8 if int(first_row["nb_byz"]) == 0 else 2.2
        ax.plot(
            ordered["round"],
            ordered["accuracy"],
            label=label,
            linewidth=linewidth,
            linestyle=line_style,
            alpha=0.95,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparaison de tous les scénarios - Accuracy")
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "courbe_comparaison_globale.png", dpi=180)
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
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required_columns = {"scenario", "round", "accuracy", "error", "aggregator", "attack", "nb_byz"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"CSV is missing columns: {', '.join(sorted(missing_columns))}")

    plot_per_scenario(df)
    plot_comparison(df)

    summary = summarize_final_round(df)
    summary.to_csv(COMPARISON_DIR / "resume_final.csv", index=False)

    print(f"\n--- Régénération Globale Terminée ---")
    print(f"Graphiques régénérés à partir de : {INPUT_CSV}")
    print(f"Dossier des graphiques par scénario : {PLOTS_DIR}")
    print(f"Courbe de comparaison globale : {COMPARISON_DIR / 'courbe_comparaison_globale.png'}")
    print(f"CSV de résumé final : {COMPARISON_DIR / 'resume_final.csv'}")


if __name__ == "__main__":
    main()