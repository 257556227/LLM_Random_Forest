from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


NUMERIC_COLUMNS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_COLUMNS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

TARGET_LABELS = {
    "good": "credit risk is low and the applicant is likely to repay",
    "bad": "credit risk is high and the applicant is more likely to default",
}


def build_credit_g_info(train_size: int, val_size: int, test_size: int) -> dict:
    return {
        "task_type": "binclass",
        "num_classes": 2,
        "n_num_features": len(NUMERIC_COLUMNS),
        "n_cat_features": len(CATEGORICAL_COLUMNS),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "task_intro": "This dataset predicts whether a credit applicant should be considered good or bad credit risk based on financial history, loan purpose, savings, employment, and personal status information.",
        "feature_intro": {
            "num": {
                "duration": "loan duration in months",
                "credit_amount": "requested credit amount",
                "installment_commitment": "installment rate as percentage of disposable income",
                "residence_since": "years at current residence",
                "age": "age in years",
                "existing_credits": "number of existing credits at this bank",
                "num_dependents": "number of people financially dependent on the applicant",
            },
            "cat": {
                "checking_status": "status of existing checking account",
                "credit_history": "credit history category",
                "purpose": "purpose of the requested credit",
                "savings_status": "status of savings account",
                "employment": "employment duration category",
                "personal_status": "personal status and sex category",
                "other_parties": "other debtors or guarantors",
                "property_magnitude": "most valuable available property category",
                "other_payment_plans": "other installment payment plans",
                "housing": "housing status",
                "job": "job skill and employment type",
                "own_telephone": "whether the applicant owns a telephone",
                "foreign_worker": "whether the applicant is a foreign worker",
            },
        },
        "target_intro": TARGET_LABELS,
        "source": "https://www.openml.org/d/31",
        "openml_id": 31,
    }


def _normalize_target(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized == "good":
        return 1
    if normalized == "bad":
        return 0
    raise ValueError(f"Unexpected credit-g target label: {value!r}")


def _clean_categorical_frame(frame):
    cleaned = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare credit-g dataset in DeLTa format")
    parser.add_argument("--output_dir", default="example_datasets/credit-g")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.35)
    parser.add_argument("--val_size_within_train", type=float, default=150 / 650)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading credit-g from OpenML...")
    X, y = fetch_openml(name="credit-g", version=1, return_X_y=True, as_frame=True)
    X = X.copy()[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = y.copy()

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

    info = build_credit_g_info(len(X_train), len(X_val), len(X_test))
    (output_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved credit-g dataset to: {output_dir}")
    print(f"Train: {N_train.shape}, Val: {N_val.shape}, Test: {N_test.shape}")


if __name__ == "__main__":
    main()
