# Import necessary libraries
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
NB_ROUNDS = 1000
DEVICE = "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = SCRIPT_DIR.parent / "test_bibliotheque/data"
OUTPUT_CSV = SCRIPT_DIR / "benchmark_results.csv"


# Repere le dossier qui contient les fichiers MNIST deja presentes dans le projet.
def resolve_data_root() -> Path:
    if DATA_CANDIDATES:
        return DATA_CANDIDATES
    raise FileNotFoundError(
        "fichier DATA introuvable"
    )


# Charge MNIST, applique la normalisation, puis decoupe les donnees en lots pour les clients et le test.
def build_dataloaders():
    data_root = resolve_data_root()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

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


# Renvoie les parametres associes a chaque agregateur robuste utilise dans l'experience.
def aggregator_parameters(name: str) -> dict:
    if name in {"TrMean", "Krum", "MultiKrum"}:
        return {"f": NB_BYZANTINE_CLIENTS}
    return {}


# Renvoie les parametres de l'attaque byzantine choisie pour le scenario courant.
def attack_parameters(name: str) -> dict:
    if name == "SignFlipping":
        return {}
    if name == "InnerProductManipulation":
        return {"tau": 2.0}
    if name == "ALittleIsEnough":
        return {"tau": 1.5}
    if name == "Gaussian":
        return {"mu": 0.0, "sigma": 1.0}
    raise ValueError(f"Unsupported attack: {name}")


# Affiche un resume rapide des attaques et agregateurs mis en jeu dans cette experience.
def print_byzfl_overview() -> None:
    print("ByzFL attacks available here:")
    print("  - SignFlipping")
    print("  - InnerProductManipulation")
    print("  - ALittleIsEnough")
    print("  - Gaussian")
    print("ByzFL aggregators used in this script:")
    print("  - Average")
    print("  - Median")
    print("  - TrMean")
    print("  - Krum")
    print()


# Construit le serveur central qui agrege les gradients et evalue le modele sur le test set.
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
                "parameters": aggregator_parameters(aggregator_name),
            },
        }
    )


# Cree un client honnete par partition de donnees locale.
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


# Execute un scenario complet et retourne les accuracies initiale et finale.
def run_scenario(name: str, aggregator_name: str, attack_name: str | None) -> dict:
    client_loaders, test_loader = build_dataloaders()
    clients = build_clients(client_loaders)
    server = build_server(test_loader, aggregator_name)

    byzantine_client = None
    if attack_name is not None:
        byzantine_client = ByzantineClient(
            {
                "name": attack_name,
                "f": NB_BYZANTINE_CLIENTS,
                "parameters": attack_parameters(attack_name),
            }
        )

    initial_accuracy = server.compute_test_accuracy()

    for _ in range(NB_ROUNDS):
        server_state = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(server_state)

        for client in clients:
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


# Sauvegarde tous les resultats compares dans un fichier CSV lisible.
def save_results(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "aggregator", "attack", "initial_accuracy", "final_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


    # Lance tous les scenarios, affiche les resultats et ecrit le CSV final.
def main() -> None:
    set_random_seed(SEED)
    torch.manual_seed(SEED)

    print_byzfl_overview()

    scenarios = [
        {"name": "clean_average", "aggregator": "Average", "attack": None},
        {"name": "signflip_average", "aggregator": "Average", "attack": "SignFlipping"},
        {"name": "signflip_median", "aggregator": "Median", "attack": "SignFlipping"},
        {"name": "ipm_trmean", "aggregator": "TrMean", "attack": "InnerProductManipulation"},
        {"name": "alie_krum", "aggregator": "Krum", "attack": "ALittleIsEnough"},
    ]

    results = []
    for scenario in scenarios:
        print(
            f"Running scenario: {scenario['name']} | aggregator={scenario['aggregator']} | attack={scenario['attack'] or 'clean'}"
        )
        result = run_scenario(scenario["name"], scenario["aggregator"], scenario["attack"])
        results.append(result)
        print(
            f"  initial_accuracy={result['initial_accuracy']:.4f} | final_accuracy={result['final_accuracy']:.4f}"
        )

    save_results(results)

    print()
    print(f"Results saved to: {OUTPUT_CSV}")
    print("Training complete.")


if __name__ == "__main__":
    main()

