from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR.parent
CODE_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

for path in (PACKAGE_DIR, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from notre_propre_distributor import (  # noqa: E402
    DATASET_SPECS,
    SingleClassDataDistributor,
)
from byzfl import DataDistributor  # noqa: E402


DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs"
NUM_CLASSES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the custom single-class distributor with a reference byzfl distributor."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Root directory containing MNIST or FashionMNIST.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset to load.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of clients to build.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used in the client loaders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for reproducible shuffling.",
    )
    parser.add_argument(
        "--reference-distribution",
        type=str,
        default="iid",
        choices=["iid", "dirichlet_niid"],
        help="Reference distribution built with byzfl.DataDistributor.",
    )
    parser.add_argument(
        "--distribution-parameter",
        type=float,
        default=0.5,
        help="Dirichlet alpha if the reference distributor is dirichlet_niid.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the CSV and PNG comparison files will be saved.",
    )
    return parser.parse_args()


def load_train_dataset(dataset_name: str, data_root: Path):
    spec = DATASET_SPECS[dataset_name]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    )

    dataset_class = spec["dataset_class"]
    return dataset_class(root=str(data_root), train=True, download=True, transform=transform)


def build_reference_loaders(
    train_dataset,
    batch_size: int,
    num_clients: int,
    distribution_name: str,
    distribution_parameter: float,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    distributor_config: dict[str, object] = {
        "data_distribution_name": distribution_name,
        "nb_honest": num_clients,
        "data_loader": train_loader,
        "batch_size": batch_size,
    }
    if distribution_name == "dirichlet_niid":
        distributor_config["distribution_parameter"] = distribution_parameter

    distributor = DataDistributor(distributor_config)
    return distributor.split_data()


def build_custom_loaders(
    train_dataset,
    batch_size: int,
    num_clients: int,
    seed: int,
):
    distributor = SingleClassDataDistributor(
        dataset=train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    return distributor.split_data(), distributor


def count_labels(loader) -> list[int]:
    counts = [0 for _ in range(NUM_CLASSES)]
    for _, labels in loader:
        for label in labels.tolist():
            counts[int(label)] += 1
    return counts


def loaders_to_dataframe(loaders, distribution_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for client_id, loader in enumerate(loaders):
        counts = count_labels(loader)
        total = sum(counts)
        dominant = max(counts) if counts else 0
        purity = float(dominant / total) if total else 0.0
        active_classes = sum(1 for value in counts if value > 0)
        for class_id, sample_count in enumerate(counts):
            rows.append(
                {
                    "distribution": distribution_name,
                    "client_id": client_id,
                    "class_id": class_id,
                    "sample_count": sample_count,
                    "total_samples": total,
                    "class_share": float(sample_count / total) if total else 0.0,
                    "purity": purity,
                    "active_classes": active_classes,
                }
            )
    return pd.DataFrame(rows)


def summarize_distribution(df: pd.DataFrame, distribution_name: str) -> None:
    client_summary = (
        df[df["distribution"] == distribution_name]
        .groupby(["client_id"], as_index=False)
        .agg(
            total_samples=("total_samples", "first"),
            purity=("purity", "first"),
            active_classes=("active_classes", "first"),
        )
    )

    print(f"\n[{distribution_name}] client summary")
    for _, row in client_summary.iterrows():
        print(
            f"Client {int(row['client_id']):02d} | samples={int(row['total_samples'])} | "
            f"purity={row['purity']:.3f} | active_classes={int(row['active_classes'])}"
        )

    print(
        f"[{distribution_name}] average purity: {client_summary['purity'].mean():.3f} | "
        f"average active classes: {client_summary['active_classes'].mean():.2f}"
    )


def plot_comparison(df: pd.DataFrame, output_path: Path) -> None:
    distributions = list(dict.fromkeys(df["distribution"].tolist()))
    num_distributions = len(distributions)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_distributions,
        figsize=(6.5 * num_distributions, 6),
        sharey=True,
        constrained_layout=True,
    )
    if num_distributions == 1:
        axes = [axes]

    max_value = int(df["sample_count"].max()) if not df.empty else 1

    last_image = None
    for axis, distribution_name in zip(axes, distributions):
        subset = df[df["distribution"] == distribution_name]
        matrix = subset.pivot(index="client_id", columns="class_id", values="sample_count")
        matrix = matrix.reindex(index=sorted(matrix.index), columns=range(NUM_CLASSES)).fillna(0)

        last_image = axis.imshow(matrix.values, aspect="auto", cmap="Blues", vmin=0, vmax=max_value)
        axis.set_title(distribution_name)
        axis.set_xlabel("Classe")
        axis.set_xticks(range(NUM_CLASSES))
        axis.set_xticklabels(range(NUM_CLASSES))
        axis.set_yticks(range(matrix.shape[0]))
        axis.set_yticklabels(matrix.index.tolist())

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = int(matrix.iat[row_index, col_index])
                if value > 0:
                    axis.text(
                        col_index,
                        row_index,
                        str(value),
                        ha="center",
                        va="center",
                        color="black" if value < max_value * 0.55 else "white",
                        fontsize=8,
                    )

    axes[0].set_ylabel("Client")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, shrink=0.85, label="Nombre d'exemples")
    fig.suptitle("Comparaison des répartitions de données", y=1.02)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_train_dataset(args.dataset, data_root)

    reference_loaders = build_reference_loaders(
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        distribution_name=args.reference_distribution,
        distribution_parameter=args.distribution_parameter,
    )
    custom_loaders, custom_distributor = build_custom_loaders(
        train_dataset=train_dataset,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    reference_df = loaders_to_dataframe(
        reference_loaders,
        distribution_name=f"byzfl_{args.reference_distribution}",
    )
    custom_df = loaders_to_dataframe(
        custom_loaders,
        distribution_name="NotrePropreDistributor",
    )
    result_df = pd.concat([reference_df, custom_df], ignore_index=True)

    csv_path = output_dir / "comparison_counts.csv"
    png_path = output_dir / "comparison_heatmap.png"
    result_df.to_csv(csv_path, index=False)
    plot_comparison(result_df, png_path)

    print("Comparison generated successfully.")
    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {png_path}")
    summarize_distribution(result_df, f"byzfl_{args.reference_distribution}")
    summarize_distribution(result_df, "NotrePropreDistributor")
    print("\nCustom distributor assignment preview:")
    custom_distributor.print_summary()


if __name__ == "__main__":
    main()