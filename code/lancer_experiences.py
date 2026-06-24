from __future__ import annotations

import argparse
import re
from pathlib import Path
import torch

# ==============================================================================
# MONKEY PATCH ULTRA-ROBUSTE POUR BYZFL (Bug as_tensor avec Python 3.11+)
# ==============================================================================
_orig_as_tensor = torch.as_tensor

def _patched_as_tensor(data, dtype=None, device=None, **kwargs):
    # Si ByzFL envoie un type qui n'est pas un type PyTorch officiel (ex: int, bool, np.float)
    if dtype is not None and not isinstance(dtype, torch.dtype):
        dtype_str = str(dtype).lower()
        if 'int' in dtype_str:
            dtype = torch.int64
        elif 'float' in dtype_str:
            dtype = torch.float32
        elif 'bool' in dtype_str:
            dtype = torch.bool
        else:
            dtype = None  # Fallback ultime : on laisse PyTorch deviner
            
    return _orig_as_tensor(data, dtype=dtype, device=device, **kwargs)

torch.as_tensor = _patched_as_tensor
# ==============================================================================

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
    # NOUVEAU PARAMÈTRE : Nombre de runs à moyenner
    parser.add_argument("--runs", type=int, default=5, help="Nombre de runs à effectuer et moyenner par scénario.")
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

    # ==========================================================================
    # MODIFICATION SÉCURITÉ : AMÉLIORATION DE LA SAUVEGARDE AU FIL DE L'EAU
    # ==========================================================================
    all_records = []
    
    # 1. On pré-charge toutes les configurations pour fixer le nom du CSV final dès le départ
    scenario_configs = []
    for entry in scenario_entries:
        scenario_configs.append(load_json(resolve_repo_path(entry["file"])))

    output_dir = resolve_repo_path(common_config.get("output_dir", "resultats_experiences"))
    output_filename = build_output_filename(scenario_configs)
    
    num_runs = args.runs  # Par défaut, 5 runs

    # 2. On exécute les scénarios un par un
    for entry, scenario_config in zip(scenario_entries, scenario_configs):
        base_config = build_config(common_config, scenario_config)
        
        print(f"\n====================================================================")
        print(f" Running scenario: {base_config['name']} (Lissage sur {num_runs} runs)")
        print(f"====================================================================")
        
        accumulated_metrics = {}
        template_records = {}
        base_seed = int(base_config.get("seed", 42))
        scenario_has_data = False

        for run_idx in range(num_runs):
            print(f"  -> Exécution du run {run_idx + 1}/{num_runs}...")
            
            run_config = base_config.copy()
            run_config["seed"] = base_seed + run_idx
            
            # Encapsulation de run_scenario pour éviter qu'un crash total d'un run ne bloque le reste
            try:
                records = run_scenario(run_config)
            except Exception as e:
                print(f"  ❌ Le run {run_idx + 1} a échoué avec l'erreur : {e}")
                records = []
            
            # On cumule les résultats round par round
            for rec in records:
                r = rec["round"]
                if r not in accumulated_metrics:
                    # Ajout d'un compteur 'count' pour savoir combien de runs ont réussi ce round
                    accumulated_metrics[r] = {"acc": 0.0, "err": 0.0, "count": 0}
                    template_records[r] = {k: v for k, v in rec.items() if k not in ["accuracy", "error"]}
                
                accumulated_metrics[r]["acc"] += rec["accuracy"]
                accumulated_metrics[r]["err"] += rec["error"]
                accumulated_metrics[r]["count"] += 1
                scenario_has_data = True

        # 3. Si le scénario a généré des données, on calcule la moyenne et on SAUVEGARDE IMMÉDIATEMENT
        if scenario_has_data:
            for r in sorted(accumulated_metrics.keys()):
                actual_runs = accumulated_metrics[r]["count"]
                if actual_runs > 0:
                    avg_record = template_records[r].copy()
                    # Moyenne intelligente basée sur le nombre réel de runs réussis pour ce round
                    avg_record["accuracy"] = accumulated_metrics[r]["acc"] / actual_runs
                    avg_record["error"] = accumulated_metrics[r]["err"] / actual_runs
                    all_records.append(avg_record)

            # === SAUVEGARDE ET GRAPHES AU FIL DE L'EAU ===
            csv_path = save_records(all_records, output_dir, filename=output_filename)
            plot_records(all_records, output_dir)
            print(f"  💾 [Sauvegarde Secours] Données cumulées écrites dans : {csv_path.name}")
            print(f"  📊 [Sauvegarde Secours] Graphiques mis à jour dans : {output_dir / 'graphiques'}")
        else:
            print(f"  ⚠️ Aucune donnée n'a pu être extraite pour le scénario {base_config['name']}.")

    print(f"\n--- TOUS LES SCÉNARIOS DISPONIBLES ONT ÉTÉ TRAITÉS ---")
    print(f"Fichier final complet : {output_dir / output_filename}")


if __name__ == "__main__":
    main()