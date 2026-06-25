from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import ByzantineClient, Client, DataDistributor, Server
from byzfl.utils.misc import set_random_seed

# ==============================================================================
# STRUCTURE DES DOSSIERS (Mise à jour Racine)
# ==============================================================================
# __file__ est dans "PROJECT_ROOT/code/moteur_experiences.py"
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Dossier des configurations au même niveau que code/
CONFIG_DIR = PROJECT_ROOT / "configurations_experiences"

DEFAULT_COMMON_CONFIG = CONFIG_DIR / "commun.json"
DEFAULT_SCENARIO_INDEX = CONFIG_DIR / "scenario_index.json"
# ==============================================================================

DATASET_SPECS = {
    "MNIST": {
        "dataset_class": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "FashionMNIST": {
        "dataset_class": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_repo_path(path_value: str | Path, *, base_path: Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_path or PROJECT_ROOT) / path


def resolve_data_root(config: dict[str, Any]) -> Path:
    data_root_value = config.get("data_root", "data")
    candidate = resolve_repo_path(data_root_value)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Data folder not found: {candidate}")


def build_dataloaders(config: dict[str, Any]):
    dataset_name = config["dataset"]
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data_root = resolve_data_root(config)
    spec = DATASET_SPECS[dataset_name]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    )

    dataset_class = spec["dataset_class"]
    train_dataset = dataset_class(
        root=str(data_root), train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root=str(data_root), train=False, download=True, transform=transform
    )

    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    distributor_config: dict[str, Any] = {
        "data_distribution_name": config.get("distribution", "iid"),
        "nb_honest": int(config["nb_honest"]),
        "data_loader": train_loader,
        "batch_size": batch_size,
    }
    if distributor_config["data_distribution_name"] == "dirichlet_niid":
        distributor_config["distribution_parameter"] = float(
            config.get("distribution_parameter", 0.5)
        )

    distributor = DataDistributor(distributor_config)
    return distributor.split_data(), test_loader


def build_aggregator_parameters(config: dict[str, Any], aggregator_name: str) -> dict[str, Any]:
    aggregator_parameters = copy.deepcopy(config.get("aggregator_parameters", {}))
    if aggregator_name in {"TrMean", "Krum", "MultiKrum", "MDA", "MoNNA", "Meamed", "CAF", "SMEA"}:
        aggregator_parameters.setdefault("f", int(config["nb_byz"]))
    return aggregator_parameters


def build_server(config: dict[str, Any], test_loader: DataLoader, aggregator_name: str, nb_byz: int) -> Server:
    aggregator_parameters = build_aggregator_parameters(config, aggregator_name)

    return Server(
        {
            "device": config.get("device", "cpu"),
            "model_name": config.get("model_name", "cnn_mnist"),
            "test_loader": test_loader,
            "optimizer_name": config.get("optimizer_name", "SGD"),
            "learning_rate": float(config.get("server_learning_rate", 0.1)),
            "weight_decay": float(config.get("server_weight_decay", 0.0001)),
            "milestones": config.get("milestones", [1000]),
            "learning_rate_decay": float(config.get("server_learning_rate_decay", 0.25)),
            "pre_agg_list": config.get("pre_agg_list", []),
            "aggregator_info": {"name": aggregator_name, "parameters": aggregator_parameters},
        }
    )


def build_clients(config: dict[str, Any], client_loaders) -> list[Client]:
    clients = []
    for loader in client_loaders:
        clients.append(
            Client(
                {
                    "model_name": config.get("model_name", "cnn_mnist"),
                    "device": config.get("device", "cpu"),
                    "optimizer_name": config.get("optimizer_name", "SGD"),
                    "learning_rate": float(config.get("client_learning_rate", 0.01)),
                    "loss_name": config.get("loss_name", "CrossEntropyLoss"),
                    "weight_decay": float(config.get("client_weight_decay", 0.0)),
                    "milestones": config.get("milestones", [1000]),
                    "learning_rate_decay": float(config.get("client_learning_rate_decay", 0.25)),
                    "LabelFlipping": False,
                    "momentum": float(config.get("client_momentum", 0.1)),
                    "store_per_client_metrics": False,
                    "training_dataloader": loader,
                    "nb_labels": int(config.get("nb_labels", 10)),
                }
            )
        )
    return clients


def attack_parameters(attack_name: str, config: dict[str, Any]) -> dict[str, Any]:
    if attack_name == "SignFlipping":
        return {}
    if attack_name == "InnerProductManipulation":
        return {"tau": float(config.get("attack_tau", 2.0))}
    if attack_name == "ALittleIsEnough":
        return {"tau": float(config.get("attack_tau", 1.5))}
    if attack_name in {"Optimal_InnerProductManipulation", "Optimal_ALittleIsEnough"}:
        return {
            "aggregator_info": {
                "name": config["aggregator"],
                "parameters": build_aggregator_parameters(config, config["aggregator"]),
            }
        }
    if attack_name in {"Mimic", "Inf"}:
        return {}
    if attack_name == "Gaussian":
        return {
            "mu": float(config.get("attack_mu", 0.0)),
            "sigma": float(config.get("attack_sigma", 1.0)),
        }
    raise ValueError(f"Unsupported attack: {attack_name}")


def run_scenario(config: dict[str, Any]) -> list[dict[str, Any]]:
    set_random_seed(int(config.get("seed", 42)))
    torch.manual_seed(int(config.get("seed", 42)))

    client_loaders, test_loader = build_dataloaders(config)
    clients = build_clients(config, client_loaders)
    server = build_server(config, test_loader, config["aggregator"], int(config["nb_byz"]))

    attack_name = config.get("attack", "none")
    byzantine_client = None
    if attack_name != "none" and int(config["nb_byz"]) > 0:
        byzantine_client = ByzantineClient(
            {
                "name": attack_name,
                "f": int(config["nb_byz"]),
                "parameters": attack_parameters(attack_name, config),
            }
        )

    records: list[dict[str, Any]] = []
    rounds = int(config.get("rounds", 100))
    eval_interval = int(config.get("eval_interval", 1))
    scenario_name = config["name"]

    for round_number in range(1, rounds + 1):
        server_state = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(server_state)
            client.compute_gradients()

        honest_gradients = [client.get_flat_gradients_with_momentum() for client in clients]
        corrupted = [] if byzantine_client is None else byzantine_client.apply_attack(honest_gradients)
        
        device = server.device if hasattr(server, 'device') else "cuda"
        corrupted = [g.to(device) for g in corrupted]

        try:
            server.update_model_with_gradients(honest_gradients + corrupted)
        except TypeError as e:
            if "'NoneType' object is not subscriptable" in str(e):
                print(f"\n [Limite Mathématique] L'agrégateur {config['aggregator']} a échoué (renvoyé None) face à l'attaque {attack_name}.")
                print("Arrêt prématuré de ce run pour passer automatiquement aux scénarios suivants.")
                break  
            else:
                raise e

        if round_number % eval_interval == 0:
            accuracy = float(server.compute_test_accuracy())
            records.append(
                {
                    "scenario": scenario_name,
                    "dataset": config["dataset"],
                    "distribution": config.get("distribution", "iid"),
                    "aggregator": config["aggregator"],
                    "attack": attack_name,
                    "nb_honest": int(config["nb_honest"]),
                    "nb_byz": int(config["nb_byz"]),
                    "round": round_number,
                    "accuracy": accuracy,
                    "error": 1.0 - accuracy,
                }
            )

    return records


def save_records(records: list[dict[str, Any]], output_dir: Path, filename: str = "resultats_combines.csv") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def plot_records(records: list[dict[str, Any]], output_dir: Path) -> None:
    plots_dir = output_dir / "graphiques"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not records:
        print("⚠️ Aucun enregistrement trouvé dans 'records'. Impossible de générer les graphiques.")
        return

    df = pd.DataFrame(records)
    
    if "scenario" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "scenario"})
        else:
            print("❌ Impossible de grouper par scénario. Génération des graphiques annulée.")
            return

    round_max = int(df["round"].max()) if not df.empty else 1
    x_max = max(round_max, 2)

    for scenario_name, group in df.groupby("scenario"):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(group["round"], group["accuracy"], color="tab:blue", linewidth=2)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title(scenario_name)
        ax.set_xlim(1, x_max)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plots_dir / f"{scenario_name}.png", dpi=150)
        plt.close(fig)


def load_common_config(common_config_path: Path = DEFAULT_COMMON_CONFIG) -> dict[str, Any]:
    return load_json(common_config_path)


def load_scenario_index(index_path: Path = DEFAULT_SCENARIO_INDEX) -> list[dict[str, Any]]:
    data = load_json(index_path)
    return data["scenarios"]


def load_scenario(scenario_entry: dict[str, Any]) -> dict[str, Any]:
    scenario_file = resolve_repo_path(scenario_entry["file"])
    return load_json(scenario_file)


def build_config(common_config: dict[str, Any], scenario_config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = deep_merge_dicts(common_config, scenario_config)
    if overrides:
        config = deep_merge_dicts(config, overrides)
    return config