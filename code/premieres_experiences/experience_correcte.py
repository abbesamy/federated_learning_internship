from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import Client, DataDistributor, Server
from byzfl.utils.misc import set_random_seed


SEED = 42
NB_HONEST_CLIENTS = 4
BATCH_SIZE = 64
NB_ROUNDS = 100
DEVICE = "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "test_bibliotheque" / "data"
OUTPUT_CSV = SCRIPT_DIR / "benchmark_results_experience_correcte.csv"


def resolve_data_root() -> Path:
    if DATA_ROOT.exists():
        return DATA_ROOT
    raise FileNotFoundError(f"Data folder not found: {DATA_ROOT}")


def build_dataloaders():
    data_root = resolve_data_root()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root=str(data_root), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_root), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    distributor = DataDistributor(
        {
            "data_distribution_name": "iid",
            "nb_honest": NB_HONEST_CLIENTS,
            "data_loader": train_loader,
            "batch_size": BATCH_SIZE,
        }
    )
    return distributor.split_data(), test_loader


def build_server(test_loader: DataLoader, aggregator_name: str) -> Server:
    return Server(
        {
            "device": DEVICE,
            "model_name": "cnn_mnist",
            "test_loader": test_loader,
            "optimizer_name": "SGD",
            "learning_rate": 0.1,
            "weight_decay": 0.0001,
            "milestones": [1000],
            "learning_rate_decay": 0.25,
            "pre_agg_list": [],
            "aggregator_info": {
                "name": aggregator_name,
                "parameters": {},
            },
        }
    )


def build_clients(client_loaders) -> list[Client]:
    clients = []
    for loader in client_loaders:
        clients.append(
            Client(
                {
                    "model_name": "cnn_mnist",
                    "device": DEVICE,
                    "optimizer_name": "SGD",
                    "learning_rate": 0.01,
                    "loss_name": "CrossEntropyLoss",
                    "weight_decay": 0.0,
                    "milestones": [1000],
                    "learning_rate_decay": 0.25,
                    "LabelFlipping": False,
                    "momentum": 0.1,
                    "store_per_client_metrics": False,
                    "training_dataloader": loader,
                    "nb_labels": 10,
                }
            )
        )
    return clients


def run_scenario(name: str, aggregator_name: str) -> dict:
    client_loaders, test_loader = build_dataloaders()
    clients = build_clients(client_loaders)
    server = build_server(test_loader, aggregator_name)

    initial_accuracy = server.compute_test_accuracy()

    for _ in range(NB_ROUNDS):
        server_state = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(server_state)

        for client in clients:
            client.compute_gradients()

        honest_gradients = [client.get_flat_gradients_with_momentum() for client in clients]
        server.update_model_with_gradients(honest_gradients)

    final_accuracy = server.compute_test_accuracy()

    return {
        "scenario": name,
        "aggregator": aggregator_name,
        "attack": "none",
        "nb_honest_clients": NB_HONEST_CLIENTS,
        "nb_byzantine_clients": 0,
        "initial_accuracy": round(float(initial_accuracy), 4),
        "final_accuracy": round(float(final_accuracy), 4),
    }


def save_results(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "aggregator",
                "attack",
                "nb_honest_clients",
                "nb_byzantine_clients",
                "initial_accuracy",
                "final_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_random_seed(SEED)
    torch.manual_seed(SEED)

    scenarios = [
        {"name": "baseline_clean_average", "aggregator": "Average"},
        {"name": "baseline_clean_median", "aggregator": "Median"},
        {"name": "baseline_clean_trmean", "aggregator": "TrMean"},
    ]

    results = []
    for scenario in scenarios:
        print(
            f"Running scenario: {scenario['name']} | aggregator={scenario['aggregator']} | attack=none"
        )
        result = run_scenario(scenario["name"], scenario["aggregator"])
        results.append(result)
        print(
            f"  initial_accuracy={result['initial_accuracy']:.4f} | final_accuracy={result['final_accuracy']:.4f}"
        )

    save_results(results)

    print()
    print(f"Results saved to: {OUTPUT_CSV}")
    print("Baseline training complete.")


if __name__ == "__main__":
    main()
