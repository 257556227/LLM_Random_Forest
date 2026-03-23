from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = [f"V{i}" for i in range(1, 55)]

TARGET_LABELS = {
    "0": "class 0 region in the anonymized high-dimensional feature space",
    "1": "class 1 region in the anonymized high-dimensional feature space",
    "2": "class 2 region in the anonymized high-dimensional feature space",
    "3": "class 3 region in the anonymized high-dimensional feature space",
}

TARGET_TO_INDEX = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}


def build_jannis_info(train_size: int, val_size: int, test_size: int) -> dict:
    return {
        "task_type": "multiclass",
        "num_classes": 4,
        "n_num_features": len(NUMERIC_COLUMNS),
        "n_cat_features": 0,
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "task_intro": "This dataset predicts one of four classes from 54 anonymized continuous features. It is a high-dimensional weak-semantic multiclass benchmark where local numeric patterns matter more than human-readable feature names.",
        "feature_intro": {
            "num": {
                column: f"anonymized continuous feature {column} from the high-dimensional jannis benchmark"
                for column in NUMERIC_COLUMNS
            },
            "cat": {},
        },
        "target_intro": TARGET_LABELS,
        "source": "https://www.openml.org/d/41168",
        "openml_id": 41168,
    }


def _normalize_target(value: object) -> int:
    normalized = str(value).strip()
    if normalized not in TARGET_TO_INDEX:
        raise ValueError(f"Unexpected jannis target label: {value!r}")
    return TARGET_TO_INDEX[normalized]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare jannis dataset in DeLTa format")
    parser.add_argument("--output_dir", default="example_datasets/jannis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size_within_train", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading jannis from OpenML...")
    X, y = fetch_openml(name="jannis", version=1, return_X_y=True, as_frame=True)
    X = X.copy()[NUMERIC_COLUMNS]
    y = y.copy()

    for column in NUMERIC_COLUMNS:
        X[column] = X[column].astype(np.float32)
    y = y.map(_normalize_target).astype(np.int64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=args.val_size_within_train,
        random_state=args.seed,
        stratify=y_trainval,
    )

    N_train = X_train[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    N_val = X_val[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    N_test = X_test[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)

    np.save(output_dir / "N_train.npy", N_train)
    np.save(output_dir / "N_val.npy", N_val)
    np.save(output_dir / "N_test.npy", N_test)
    np.save(output_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(output_dir / "y_val.npy", y_val.to_numpy(dtype=np.int64))
    np.save(output_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))

    info = build_jannis_info(len(X_train), len(X_val), len(X_test))
    (output_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved jannis dataset to: {output_dir}")
    print(f"Train: {N_train.shape}, Val: {N_val.shape}, Test: {N_test.shape}")


if __name__ == "__main__":
    main()