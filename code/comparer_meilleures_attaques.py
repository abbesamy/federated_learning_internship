from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "resultats_experiences"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "comparaison_finale"

ATTACK_ORDER = [
    "SignFlipping",
    "InnerProductManipulation",
    "Optimal_InnerProductManipulation",
    "ALittleIsEnough",
    "Optimal_ALittleIsEnough",
    "Mimic",
    "Inf",
    "Gaussian",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the best aggregator curve for each attack and rank attacks by final accuracy."
    )
    parser.add_argument(
        "--input-csv",
        nargs="*",
        type=str,
        help="One or more results CSV files. If omitted, the script auto-discovers available results CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the final comparison PNG and CSVs will be written.",
    )
    return parser.parse_args()


def sanitize_name(value: object) -> str:
    value_str = str(value) if value is not None else ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value_str.strip().lower())
    return normalized.strip("_") or "attaque"


def discover_input_csvs() -> list[Path]:
    preferred_names = [
        "resultats_combines.csv",
        "resultats_colab.csv",
        "resultats_colab_produit_interne.csv",
    ]
    discovered = [RESULTS_DIR / name for name in preferred_names if (RESULTS_DIR / name).exists()]
    if discovered:
        return discovered
    return sorted(RESULTS_DIR.glob("resultats*.csv"))


def ordered_attacks(df: pd.DataFrame) -> list[str]:
    discovered = list(dict.fromkeys(df["attack"].tolist()))
    ordered = [name for name in ATTACK_ORDER if name in discovered]
    ordered.extend(sorted(name for name in discovered if name not in ordered))
    return ordered


def build_attack_color_map(attacks: list[str]) -> dict[str, str]:
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    return {attack: palette[index % len(palette)] for index, attack in enumerate(attacks)}


def summarize_best_aggregator_per_attack(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.dropna(subset=["attack", "scenario", "aggregator", "round", "accuracy"]).copy()
    clean_df["attack"] = clean_df["attack"].astype(str)
    clean_df["scenario"] = clean_df["scenario"].astype(str)
    clean_df["aggregator"] = clean_df["aggregator"].astype(str)

    final_rounds = clean_df.sort_values("round").groupby(["attack", "scenario"], as_index=False).tail(1)
    best_by_attack = final_rounds.sort_values(["attack", "accuracy"], ascending=[True, False])
    best_by_attack = best_by_attack.groupby("attack", as_index=False).head(1).copy()
    best_by_attack = best_by_attack[
        ["attack", "scenario", "aggregator", "round", "accuracy", "error", "nb_honest", "nb_byz"]
    ].reset_index(drop=True)
    best_by_attack.insert(0, "rank_dans_attaque", best_by_attack.groupby("attack").cumcount() + 1)
    best_by_attack = best_by_attack.sort_values("attack").reset_index(drop=True)
    return best_by_attack


def build_attack_ranking(best_by_attack: pd.DataFrame) -> pd.DataFrame:
    ranking = best_by_attack.sort_values(["accuracy", "attack"], ascending=[True, True]).reset_index(drop=True)
    ranking = ranking.copy()
    ranking.insert(0, "rank_pire_attaque", range(1, len(ranking) + 1))
    return ranking[
        [
            "rank_pire_attaque",
            "attack",
            "scenario",
            "aggregator",
            "round",
            "accuracy",
            "error",
            "nb_honest",
            "nb_byz",
        ]
    ]


def extract_selected_curves(df: pd.DataFrame, best_by_attack: pd.DataFrame) -> pd.DataFrame:
    selected_frames = []
    for _, row in best_by_attack.iterrows():
        curve = df[df["scenario"] == row["scenario"]].copy()
        curve["selected_attack"] = row["attack"]
        curve["selected_aggregator"] = row["aggregator"]
        selected_frames.append(curve)
    if not selected_frames:
        return pd.DataFrame()
    return pd.concat(selected_frames, ignore_index=True).sort_values(["selected_attack", "round"])


def plot_selected_curves(df_selected: pd.DataFrame, output_path: Path, color_map: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    round_max = int(df_selected["round"].max()) if not df_selected.empty else 1
    x_max = max(round_max, 2)

    for attack_name, group in df_selected.groupby("selected_attack", sort=False):
        ordered = group.sort_values("round")
        chosen_aggregator = ordered.iloc[0]["selected_aggregator"]
        ax.plot(
            ordered["round"],
            ordered["accuracy"],
            linewidth=2.6,
            color=color_map.get(attack_name),
            label=f"{attack_name} | {chosen_aggregator}",
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparaison finale des meilleures courbes par attaque")
    ax.set_xlim(1, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.input_csv:
        input_csvs = [Path(value) for value in args.input_csv]
    else:
        input_csvs = discover_input_csvs()

    input_csvs = [path for path in input_csvs if path.exists()]
    if not input_csvs:
        raise SystemExit(f"Aucun CSV de résultats trouvé dans : {RESULTS_DIR}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(path) for path in input_csvs]
    df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    required_columns = {"scenario", "round", "accuracy", "error", "aggregator", "attack", "nb_byz"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"CSV incomplet, colonnes manquantes : {', '.join(sorted(missing_columns))}")

    best_by_attack = summarize_best_aggregator_per_attack(df)
    ranking = build_attack_ranking(best_by_attack)
    selected_curves = extract_selected_curves(df, best_by_attack)

    best_csv = output_dir / "meilleur_agregateur_par_attaque.csv"
    best_by_attack.to_csv(best_csv, index=False)

    ranking_csv = output_dir / "classement_final_par_attaque.csv"
    ranking.to_csv(ranking_csv, index=False)

    selected_curves_csv = output_dir / "courbes_selectionnees_par_attaque.csv"
    selected_curves.to_csv(selected_curves_csv, index=False)

    attacks = ordered_attacks(best_by_attack)
    color_map = build_attack_color_map(attacks)

    plot_path = output_dir / "courbe_comparaison_meilleures_attaques.png"
    plot_selected_curves(selected_curves, plot_path, color_map)

    print(f"\n--- Comparaison Finale Terminée ---")
    print(f"Graphique final : {plot_path}")
    print(f"CSV des meilleurs agrégateurs par attaque : {best_csv}")
    print(f"CSV du classement final des attaques : {ranking_csv}")
    print(f"CSV des courbes sélectionnées : {selected_curves_csv}")
    print(f"CSV source(s) utilisées : {', '.join(str(path) for path in input_csvs)}")
    print("\nClassement des attaques selon leur meilleure courbe finale :")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()