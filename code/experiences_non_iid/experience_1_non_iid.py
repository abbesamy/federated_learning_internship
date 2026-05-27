from __future__ import annotations
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from byzfl import ByzantineClient, Client, DataDistributor, Server
from byzfl.utils.misc import set_random_seed

SEED = 42
NB_HONEST_CLIENTS = 3
NB_BYZANTINE_CLIENTS = 1
BATCH_SIZE = 64
NB_ROUNDS = 100
DEVICE = "cpu"

# gestion des chemins
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parents[1] / "data"
OUTPUT_CSV = SCRIPT_DIR / "benchmark_results.csv"

def build_dataloaders():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=str(DATA_ROOT), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    distributor = DataDistributor({
        "data_distribution_name": "dirichlet_niid",
        "distribution_parameter": 0.5,
        "nb_honest": NB_HONEST_CLIENTS,
        "data_loader": train_loader,
        "batch_size": BATCH_SIZE,
    })
    return distributor.split_data(), test_loader

def aggregator_parameters(name: str) -> dict:
    if name in {"TrMean", "Krum", "MultiKrum"}:
        return {"f": NB_BYZANTINE_CLIENTS}
    return {}

def attack_parameters(name: str) -> dict:
    if name == "SignFlipping": return {}
    if name == "InnerProductManipulation": return {"tau": 2.0}
    if name == "ALittleIsEnough": return {"tau": 1.5}
    if name == "Gaussian": return {"mu": 0.0, "sigma": 1.0}
    raise ValueError(f"Unsupported attack: {name}")

def build_server(test_loader: DataLoader, aggregator_name: str) -> Server:
    return Server({
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
            "parameters": aggregator_parameters(aggregator_name),
        },
    })

def build_clients(client_loaders) -> list[Client]:
    return [
        Client({
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
        }) for loader in client_loaders
    ]

def run_scenario(name: str, aggregator_name: str, attack_name: str | None) -> dict:
    client_loaders, test_loader = build_dataloaders()
    clients = build_clients(client_loaders)
    server = build_server(test_loader, aggregator_name)

    byzantine_client = None
    if attack_name is not None:
        byzantine_client = ByzantineClient({
            "name": attack_name,
            "f": NB_BYZANTINE_CLIENTS,
            "parameters": attack_parameters(attack_name),
        })

    initial_accuracy = server.compute_test_accuracy()

    for _ in range(NB_ROUNDS):
        server_state = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(server_state)
            client.compute_gradients()

        honest_gradients = [client.get_flat_gradients_with_momentum() for client in clients]
        corrupted_gradients = [] if byzantine_client is None else byzantine_client.apply_attack(honest_gradients)
        server.update_model_with_gradients(honest_gradients + corrupted_gradients)

    final_accuracy = server.compute_test_accuracy()

    return {
        "scenario": name,
        "aggregator": aggregator_name,
        "attack": attack_name or "clean",
        "initial_accuracy": round(float(initial_accuracy), 4),
        "final_accuracy": round(float(final_accuracy), 4),
    }

def save_results(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "aggregator", "attack", "initial_accuracy", "final_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    set_random_seed(SEED)
    torch.manual_seed(SEED)

    scenarios = [
        {"name": "clean_average", "aggregator": "Average", "attack": None},
        {"name": "signflip_average", "aggregator": "Average", "attack": "SignFlipping"},
        {"name": "signflip_median", "aggregator": "Median", "attack": "SignFlipping"},
        {"name": "ipm_trmean", "aggregator": "TrMean", "attack": "InnerProductManipulation"},
        {"name": "alie_krum", "aggregator": "Krum", "attack": "ALittleIsEnough"},
    ]

    results = []
    for scenario in scenarios:
        print(f"Running: {scenario['name']}...")
        result = run_scenario(scenario["name"], scenario["aggregator"], scenario["attack"])
        results.append(result)
        print(f"  initial_acc={result['initial_accuracy']:.4f} | final_acc={result['final_accuracy']:.4f}")

    save_results(results)

if __name__ == "__main__":
    main()