from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class ClientClassAssignment:
    client_id: int
    class_id: int
    sample_count: int


class SingleClassDataDistributor:
    """Split a labeled dataset so that each client receives data from one class only.

    The default behavior is one class per client, in order:
    client 0 -> class 0, client 1 -> class 1, ..., client 9 -> class 9.

    This is useful to create a very non-iid setting where every client learns
    only one label.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int = 10,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
        client_classes: Sequence[int] | None = None,
        drop_last: bool = False,
    ) -> None:
        if num_clients <= 0:
            raise ValueError("num_clients must be strictly positive")
        self.dataset = dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if client_classes is None:
            if num_clients > 10:
                raise ValueError(
                    "By default this distributor expects at most 10 clients, one per class."
                )
            client_classes = list(range(num_clients))
        if len(client_classes) != num_clients:
            raise ValueError("client_classes must have exactly one class per client")

        self.client_classes = list(client_classes)
        self._targets = self._extract_targets(dataset)
        self._indices_by_class = self._group_indices_by_class(self._targets)

    @staticmethod
    def _extract_targets(dataset: Dataset) -> list[int]:
        if hasattr(dataset, "targets"):
            targets = getattr(dataset, "targets")
        elif hasattr(dataset, "labels"):
            targets = getattr(dataset, "labels")
        else:
            raise AttributeError("The dataset must expose `targets` or `labels`.")

        if isinstance(targets, torch.Tensor):
            return targets.tolist()
        return [int(value) for value in list(targets)]

    @staticmethod
    def _group_indices_by_class(targets: Sequence[int]) -> dict[int, list[int]]:
        indices_by_class: dict[int, list[int]] = {}
        for index, label in enumerate(targets):
            indices_by_class.setdefault(int(label), []).append(index)
        return indices_by_class

    def split_data(self) -> list[DataLoader]:
        generators = [torch.Generator().manual_seed(self.seed + client_id) for client_id in range(self.num_clients)]
        loaders: list[DataLoader] = []
        for client_id, class_id in enumerate(self.client_classes):
            class_indices = list(self._indices_by_class.get(int(class_id), []))
            if not class_indices:
                raise ValueError(f"No sample found for class {class_id}.")

            if self.shuffle:
                class_indices = torch.tensor(class_indices)[torch.randperm(len(class_indices), generator=generators[client_id])].tolist()

            subset = Subset(self.dataset, class_indices)
            loaders.append(
                DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last,
                    generator=generators[client_id],
                )
            )
        return loaders

    def assignment_summary(self) -> list[ClientClassAssignment]:
        summary: list[ClientClassAssignment] = []
        for client_id, class_id in enumerate(self.client_classes):
            count = len(self._indices_by_class.get(int(class_id), []))
            summary.append(
                ClientClassAssignment(
                    client_id=client_id,
                    class_id=int(class_id),
                    sample_count=count,
                )
            )
        return summary

    def print_summary(self) -> None:
        print("Client -> class assignment")
        for item in self.assignment_summary():
            print(f"Client {item.client_id:02d} -> class {item.class_id} ({item.sample_count} samples)")


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


def build_mnist_one_class_loaders(
    data_root: str | Path,
    dataset_name: str = "MNIST",
    num_clients: int = 10,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[list[DataLoader], DataLoader, SingleClassDataDistributor]:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    spec = DATASET_SPECS[dataset_name]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    )

    dataset_class = spec["dataset_class"]
    train_dataset = dataset_class(root=str(data_root), train=True, download=True, transform=transform)
    test_dataset = dataset_class(root=str(data_root), train=False, download=True, transform=transform)

    distributor = SingleClassDataDistributor(
        dataset=train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    return distributor.split_data(), DataLoader(test_dataset, batch_size=batch_size, shuffle=False), distributor
