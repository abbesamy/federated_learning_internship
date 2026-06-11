from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from distributeur_une_classe_par_client import build_mnist_one_class_loaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "MNIST"


def main() -> None:
    client_loaders, test_loader, distributor = build_mnist_one_class_loaders(
        data_root=DATA_ROOT,
        dataset_name="MNIST",
        num_clients=10,
        batch_size=64,
        seed=42,
    )

    distributor.print_summary()
    print(f"\nNumber of client loaders: {len(client_loaders)}")
    print(f"Test batches available: {len(test_loader)}")

    first_batch = next(iter(client_loaders[0]))
    labels = first_batch[1]
    print(f"First client batch labels: {sorted(set(labels.tolist()))}")


if __name__ == "__main__":
    main()
