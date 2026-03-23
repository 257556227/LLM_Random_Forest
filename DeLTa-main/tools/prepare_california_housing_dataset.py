"""
自动下载并准备 california_housing 数据集，生成 N_train.npy/N_val.npy/N_test.npy/y_train.npy/y_val.npy/y_test.npy
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "example_datasets" / "california_housing"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 下载数据
data = fetch_california_housing()
X = data.data
y = data.target

# 先分 train+val/test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=4128, random_state=42
)
# 再分 train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=3303, random_state=42
)

np.save(DATA_DIR / "N_train.npy", X_train)
np.save(DATA_DIR / "N_val.npy", X_val)
np.save(DATA_DIR / "N_test.npy", X_test)
np.save(DATA_DIR / "y_train.npy", y_train)
np.save(DATA_DIR / "y_val.npy", y_val)
np.save(DATA_DIR / "y_test.npy", y_test)

print("california_housing 数据集已准备完毕，文件已保存到:", DATA_DIR)
