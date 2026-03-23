import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST in DeLTa dataset format")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="example_datasets/mnist",
        help="Output directory for MNIST dataset in DeLTa format",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for val split")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MNIST from OpenML...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int64)

    # Use standard MNIST split: 60k train / 10k test
    X_train_all, X_test = X[:60000], X[60000:]
    y_train_all, y_test = y[:60000], y[60000:]

    # Split validation from train (50k train / 10k val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=10000,
        random_state=args.seed,
        stratify=y_train_all,
    )

    # Normalize pixel values to [0, 1]
    X_train = (X_train.astype(np.float32) / 255.0)
    X_val = (X_val.astype(np.float32) / 255.0)
    X_test = (X_test.astype(np.float32) / 255.0)

    np.save(output_dir / "N_train.npy", X_train)
    np.save(output_dir / "N_val.npy", X_val)
    np.save(output_dir / "N_test.npy", X_test)

    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)

    info = {
        "task_type": "multiclass",
        "n_num_features": 784,
        "n_cat_features": 0,
    }

    with open(output_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"Saved MNIST dataset to: {output_dir}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()
