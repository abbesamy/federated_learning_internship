from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STRUCTURE DES DOSSIERS (Mise à jour Racine)
# ==============================================================================
# __file__ est dans "PROJECT_ROOT/code/comparer_attaques.py" (ou équivalent)
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Dossier des résultats globaux sorti de code/ et placé au même niveau
RESULTS_DIR = PROJECT_ROOT / "resultats_experiences"

# Dossier par défaut pour les graphiques comparatifs
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "comparaison_par_attaque"
# ==============================================================================

AGGREGATOR_ORDER = ["Average", "Median", "TrMean", "Krum", "MultiKrum", "MoNNA", "CenteredClipping"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparer les courbes d'accuracy par attaque, avec une couleur par agrégateur."
    )
    parser.add_argument(
        "--input-csv",
        nargs="*",
        type=str,
        help="Un ou plusieurs CSV de résultats. Si absent, le script prend tous les CSV de résultats disponibles.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Dossier où écrire les PNG et le CSV de synthèse.",
    )
    return parser.parse_args()


def sanitize_name(value: any) -> str:
    value_str = str(value) if value is not None else ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value_str.strip().lower())
    return normalized.strip("_") or "attaque"


def build_color_map(aggregators: list[str]) -> dict[str, str]:
    palette = list(plt.get_cmap("tab20").colors)
    color_map: dict[str, str] = {}
    for index, aggregator in enumerate(aggregators):
        color_map[aggregator] = palette[index % len(palette)]
    return color_map


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


def ordered_aggregators(df: pd.DataFrame) -> list[str]:
    discovered = list(dict.fromkeys(df["aggregator"].tolist()))
    ordered = [name for name in AGGREGATOR_ORDER if name in discovered]
    ordered.extend(sorted(name for name in discovered if name not in ordered))
    return ordered


def plot_attack_curves(
    df_attack: pd.DataFrame,
    attack_name: str,
    output_path: Path,
    color_map: dict[str, str],
    round_max: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    attack_groups = list(df_attack.groupby("aggregator", sort=False))

    for aggregator_name, group in attack_groups:
        ordered = group.sort_values("round")
        ax.plot(
            ordered["round"],
            ordered["accuracy"],
            label=aggregator_name,
            linewidth=2.4,
            color=color_map.get(aggregator_name),
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Attaque : {attack_name}")
    ax.set_xlim(1, max(round_max, 2))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_attack(df: pd.DataFrame) -> pd.DataFrame:
    final_rounds = df.sort_values("round").groupby(["attack", "aggregator"], as_index=False).tail(1)
    summary = final_rounds[
        ["attack", "scenario", "aggregator", "round", "accuracy", "error", "nb_honest", "nb_byz"]
    ].copy()
    summary = summary.sort_values(["attack", "accuracy"], ascending=[True, False]).reset_index(drop=True)
    summary.insert(0, "rank_dans_attaque", summary.groupby("attack").cumcount() + 1)
    return summary


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

    round_max = int(df["round"].max()) if not df.empty else 1
    aggregators = ordered_aggregators(df)
    color_map = build_color_map(aggregators)

    summary = summarize_attack(df)
    summary_csv = output_dir / "resume_par_attaque.csv"
    summary.to_csv(summary_csv, index=False)

    for attack_name, df_attack in df.groupby("attack", sort=False):
        plot_path = output_dir / f"courbe_{sanitize_name(attack_name)}_agregateurs.png"
        plot_attack_curves(df_attack, attack_name, plot_path, color_map, round_max)
        print(f"Courbe enregistrée : {plot_path}")

    best_by_attack = summary.sort_values(["attack", "accuracy"], ascending=[True, False]).groupby("attack", as_index=False).head(1)
    best_csv = output_dir / "meilleur_agregateur_par_attaque.csv"
    best_by_attack.to_csv(best_csv, index=False)

    print(f"\n--- Synthèse Terminée ---")
    print(f"CSV de synthèse : {summary_csv}")
    print(f"CSV du meilleur agrégateur par attaque : {best_csv}")
    print(f"CSV source(s) utilisées : {', '.join(str(path) for path in input_csvs)}")
    print("\nMeilleurs agrégateurs par attaque :")
    print(best_by_attack.to_string(index=False))


if __name__ == "__main__":
    main()