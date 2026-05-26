from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed

import matplotlib.pyplot as plt
import pandas as pd


SEED = 42
DEVICE = "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parents[1] / "data"
OUTPUT_DIR = SCRIPT_DIR / "experiments_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_data_root() -> Path:
    if DATA_ROOT.exists():
        return DATA_ROOT
    raise FileNotFoundError(f"Data folder not found: {DATA_ROOT}")


def build_dataloaders(batch_size: int, nb_honest: int):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    distributor = DataDistributor(
        {
            "data_distribution_name": "iid",
            "nb_honest": nb_honest,
            "data_loader": train_loader,
            "batch_size": batch_size,
        }
    )
    return distributor.split_data(), test_loader


def build_server(test_loader: DataLoader, aggregator_name: str, nb_byz: int) -> Server:
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
            "aggregator_info": {"name": aggregator_name, "parameters": {"f": nb_byz}},
        }
    )


def build_clients(client_loaders) -> List[Client]:
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


def run_one_experiment(
    name: str,
    aggregator: str,
    attack: str | None,
    nb_honest: int,
    nb_byz: int,
    nb_rounds: int,
    batch_size: int,
    eval_interval: int = 1,
):
    client_loaders, test_loader = build_dataloaders(batch_size, nb_honest)
    clients = build_clients(client_loaders)
    server = build_server(test_loader, aggregator, nb_byz)

    byzantine_client = None
    if attack is not None and nb_byz > 0:
        # prepare ByzantineClient with parameters left to defaults; more customization can be added
        byzantine_client = ByzantineClient({"name": attack, "f": nb_byz, "parameters": {}})

    records = []

    for r in range(1, nb_rounds + 1):
        server_state = server.get_dict_parameters()
        for client in clients:
            client.set_model_state(server_state)

        for client in clients:
            client.compute_gradients()

        honest_gradients = [client.get_flat_gradients_with_momentum() for client in clients]
        corrupted = [] if byzantine_client is None else byzantine_client.apply_attack(honest_gradients)

        server.update_model_with_gradients(honest_gradients + corrupted)

        if r % eval_interval == 0:
            acc = server.compute_test_accuracy()
            records.append(
                {
                    "scenario": name,
                    "aggregator": aggregator,
                    "attack": attack or "clean",
                    "nb_honest": nb_honest,
                    "nb_byz": nb_byz,
                    "round": r,
                    "accuracy": float(acc),
                    "error": 1.0 - float(acc),
                }
            )

    return records


def run_grid(
    aggregators: List[str],
    attacks: List[str | None],
    nb_honest: int,
    byz_list: List[int],
    nb_rounds: int,
    batch_size: int,
    eval_interval: int,
):
    all_records = []
    for agg in aggregators:
        for attack in attacks:
            for nb_byz in byz_list:
                name = f"{agg}_{attack or 'clean'}_{nb_byz}byz"
                print(f"Running: {name}")
                recs = run_one_experiment(name, agg, attack, nb_honest, nb_byz, nb_rounds, batch_size, eval_interval)
                all_records.extend(recs)
                # save interim results
                df = pd.DataFrame(all_records)
                csv_path = OUTPUT_DIR / "experiments_results.csv"
                df.to_csv(csv_path, index=False)
    return all_records


def plot_results(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    for (aggregator, attack), group in df.groupby(["aggregator", "attack"]):
        plt.figure()
        for nb_byz, sub in group.groupby("nb_byz"):
            plt.plot(sub["round"], sub["accuracy"], label=f"{nb_byz} byz")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title(f"{aggregator} - {attack}")
        plt.legend()
        out_file = output_dir / f"plot_{aggregator}_{attack}.png"
        plt.savefig(out_file)
        plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--nb-honest", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--byz-list", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--aggregators", nargs="+", default=["Average", "Median", "TrMean"]) 
    p.add_argument("--attacks", nargs="+", default=["none", "SignFlipping", "InnerProductManipulation"]) 
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(SEED)
    torch.manual_seed(SEED)

    attacks = [None if a == "none" else a for a in args.attacks]

    records = run_grid(
        aggregators=args.aggregators,
        attacks=attacks,
        nb_honest=args.nb_honest,
        byz_list=args.byz_list,
        nb_rounds=args.rounds,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
    )

    csv_path = OUTPUT_DIR / "experiments_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    plot_results(csv_path, OUTPUT_DIR)

    print("Experiments complete. Results and plots in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
