from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


CATEGORICAL_COLUMNS = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
]

TARGET_LABELS = {
    "unacc": "the car is unacceptable for purchase under the current attribute combination",
    "acc": "the car is acceptable but not considered a strong recommendation",
    "good": "the car is considered good and meets the buyer's needs well",
    "vgood": "the car is considered very good and stands out across the evaluated criteria",
}

TARGET_TO_INDEX = {
    "unacc": 0,
    "acc": 1,
    "good": 2,
    "vgood": 3,
}


def build_car_info(train_size: int, val_size: int, test_size: int) -> dict:
    return {
        "task_type": "multiclass",
        "num_classes": 4,
        "n_num_features": 0,
        "n_cat_features": len(CATEGORICAL_COLUMNS),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "task_intro": "This dataset predicts the quality evaluation of a car based on buying price, maintenance cost, door count, passenger capacity, luggage boot size, and safety level.",
        "feature_intro": {
            "num": {},
            "cat": {
                "buying": "buying price category of the car",
                "maint": "maintenance cost category of the car",
                "doors": "number of doors category",
                "persons": "passenger capacity category",
                "lug_boot": "luggage boot size category",
                "safety": "safety rating category",
            },
        },
        "target_intro": TARGET_LABELS,
        "source": "https://www.openml.org/d/21",
        "openml_id": 21,
    }


def _normalize_target(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized not in TARGET_TO_INDEX:
        raise ValueError(f"Unexpected car target label: {value!r}")
    return TARGET_TO_INDEX[normalized]


def _clean_categorical_frame(frame):
    cleaned = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare car dataset in DeLTa format")
    parser.add_argument("--output_dir", default="example_datasets/car")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=608 / 1728)
    parser.add_argument("--val_size_within_train", type=float, default=256 / 1120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading car from OpenML...")
    X, y = fetch_openml(name="car", version=1, return_X_y=True, as_frame=True)
    X = X.copy()[CATEGORICAL_COLUMNS]
    y = y.copy()

    X = _clean_categorical_frame(X)
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

    C_train = X_train[CATEGORICAL_COLUMNS].to_numpy(dtype=object)
    C_val = X_val[CATEGORICAL_COLUMNS].to_numpy(dtype=object)
    C_test = X_test[CATEGORICAL_COLUMNS].to_numpy(dtype=object)

    np.save(output_dir / "C_train.npy", C_train)
    np.save(output_dir / "C_val.npy", C_val)
    np.save(output_dir / "C_test.npy", C_test)
    np.save(output_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(output_dir / "y_val.npy", y_val.to_numpy(dtype=np.int64))
    np.save(output_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))

    info = build_car_info(len(X_train), len(X_val), len(X_test))
    (output_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved car dataset to: {output_dir}")
    print(f"Train: {C_train.shape}, Val: {C_val.shape}, Test: {C_test.shape}")


if __name__ == "__main__":
    main()