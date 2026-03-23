from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_LABELS = {
    ">50K": "income is less than or equal to 50K per year",
    "<=50K": "income is greater than 50K per year",
    "income is greater than 50K per year": "income is less than or equal to 50K per year",
    "income is less than or equal to 50K per year": "income is greater than 50K per year",
}


def build_adult_info(train_size: int, val_size: int, test_size: int) -> dict:
    return {
        "task_type": "binclass",
        "num_classes": 2,
        "n_num_features": len(NUMERIC_COLUMNS),
        "n_cat_features": len(CATEGORICAL_COLUMNS),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "task_intro": "This dataset predicts whether a person's income exceeds 50K per year based on census and demographic attributes.",
        "feature_intro": {
            "num": {
                "age": "age in years",
                "fnlwgt": "final sampling weight",
                "education-num": "number of years of education",
                "capital-gain": "capital gains in the year",
                "capital-loss": "capital losses in the year",
                "hours-per-week": "hours worked per week",
            },
            "cat": {
                "workclass": "employment type or work class",
                "education": "highest education category",
                "marital-status": "marital status category",
                "occupation": "occupation type",
                "relationship": "household relationship status",
                "race": "race category",
                "sex": "sex category",
                "native-country": "country of origin",
            },
        },
        "target_intro": TARGET_LABELS,
        "source": "https://www.openml.org/d/1590",
        "openml_id": 1590,
    }


def _normalize_target(value: object) -> int:
    normalized = str(value).strip().rstrip(".")
    if normalized == ">50K":
        return 1
    if normalized == "<=50K":
        return 0
    raise ValueError(f"Unexpected adult target label: {value!r}")


def _clean_categorical_frame(frame):
    cleaned = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Adult dataset in DeLTa format")
    parser.add_argument("--output_dir", default="example_datasets/adult")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=1 / 3)
    parser.add_argument("--val_size_within_train", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Adult from OpenML...")
    X, y = fetch_openml(data_id=1590, return_X_y=True, as_frame=True)
    X = X.copy()
    y = y.copy()

    X = X[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    for column in NUMERIC_COLUMNS:
        X[column] = X[column].astype(float)
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

    N_train = X_train[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    N_val = X_val[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)
    N_test = X_test[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)

    C_train = X_train[CATEGORICAL_COLUMNS].to_numpy(dtype=object)
    C_val = X_val[CATEGORICAL_COLUMNS].to_numpy(dtype=object)
    C_test = X_test[CATEGORICAL_COLUMNS].to_numpy(dtype=object)

    np.save(output_dir / "N_train.npy", N_train)
    np.save(output_dir / "N_val.npy", N_val)
    np.save(output_dir / "N_test.npy", N_test)
    np.save(output_dir / "C_train.npy", C_train)
    np.save(output_dir / "C_val.npy", C_val)
    np.save(output_dir / "C_test.npy", C_test)
    np.save(output_dir / "y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save(output_dir / "y_val.npy", y_val.to_numpy(dtype=np.int64))
    np.save(output_dir / "y_test.npy", y_test.to_numpy(dtype=np.int64))

    info = build_adult_info(len(X_train), len(X_val), len(X_test))
    (output_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved Adult dataset to: {output_dir}")
    print(f"Train: {N_train.shape}, Val: {N_val.shape}, Test: {N_test.shape}")


if __name__ == "__main__":
    main()
