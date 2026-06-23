from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STRUCTURE DES DOSSIERS (Mise à jour Racine)
# ==============================================================================
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Dossier des résultats globaux
RESULTS_DIR = PROJECT_ROOT / "resultats_experiences"

# Dossier par défaut pour les graphiques comparatifs
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "comparaison_par_attaque"
# ==============================================================================

AGGREGATOR_ORDER = ["Average", "Median", "TrMean", "Krum", "MultiKrum", "MoNNA", "CenteredClipping"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparer les courbes d'accuracy par attaque et distribution, avec une couleur par agrégateur."
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
    all_discovered = sorted(RESULTS_DIR.glob("resultats*.csv"))
    return all_discovered


def ordered_aggregators(df: pd.DataFrame) -> list[str]:
    discovered = list(dict.fromkeys(df["aggregator"].tolist()))
    ordered = [name for name in AGGREGATOR_ORDER if name in discovered]
    ordered.extend(sorted(name for name in discovered if name not in ordered))
    return ordered


def plot_attack_curves(
    df_attack: pd.DataFrame,
    title_name: str,
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
    ax.set_title(title_name)
    ax.set_xlim(1, max(round_max, 2))  # S'adaptera dynamiquement au max du groupe !
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_attack(df: pd.DataFrame) -> pd.DataFrame:
    # Le .tail(1) va s'adapter automatiquement : il prendra le round 500 ou 200 selon ce qui existe pour ce groupe
    final_rounds = df.sort_values("round").groupby(["attack", "distribution", "aggregator"], as_index=False).tail(1)
    summary = final_rounds[
        ["attack", "distribution", "scenario", "aggregator", "round", "accuracy", "error", "nb_honest", "nb_byz"]
    ].copy()
    summary = summary.sort_values(["attack", "distribution", "accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    summary.insert(0, "rank_dans_contexte", summary.groupby(["attack", "distribution"]).cumcount() + 1)
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
    required_columns = {"scenario", "distribution", "round", "accuracy", "error", "aggregator", "attack", "nb_byz"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"CSV incomplet, colonnes manquantes : {', '.join(sorted(missing_columns))}")

    aggregators = ordered_aggregators(df)
    color_map = build_color_map(aggregators)

    summary = summarize_attack(df)
    summary_csv = output_dir / "resume_par_attaque.csv"
    summary.to_csv(summary_csv, index=False)

    # On boucle sur chaque couple (attack, distribution)
    for (attack_name, dist_name), df_group in df.groupby(["attack", "distribution"], sort=False):
        clean_attack = sanitize_name(attack_name)
        clean_dist = sanitize_name(dist_name)
        
        # === LA MAGIE SE PASSE ICI ===
        # Au lieu d'une limite globale, on regarde le round max UNIQUE à ce graphique précis.
        group_round_max = int(df_group["round"].max()) if not df_group.empty else 1
        # =============================
        
        plot_path = output_dir / f"courbe_{clean_attack}_{clean_dist}_agregateurs.png"
        title = f"Attaque : {attack_name} ({dist_name})"
        
        # On passe la limite sur mesure à la fonction de dessin
        plot_attack_curves(df_group, title, plot_path, color_map, group_round_max)
        print(f"Courbe enregistrée : {plot_path} (Axe X calé sur {group_round_max} rounds)")

    best_by_context = summary.sort_values(["attack", "distribution", "accuracy"], ascending=[True, True, False]).groupby(["attack", "distribution"], as_index=False).head(1)
    best_csv = output_dir / "meilleur_agregateur_par_attaque.csv"
    best_by_context.to_csv(best_csv, index=False)

    print(f"\n--- Synthèse Terminée ---")
    print(f"CSV de synthèse : {summary_csv}")
    print(f"CSV du meilleur agrégateur par contexte : {best_csv}")
    print(f"CSV source(s) utilisées : {', '.join(str(path) for path in input_csvs)}")
    print("\nMeilleurs agrégateurs par contexte (Attaque + Distribution) :")
    print(best_by_context.to_string(index=False))


if __name__ == "__main__":
    main()