#!/usr/bin/env python3
"""
下载并生成所有需要的数据集
"""
import os
import numpy as np

# 数据集目录
BASE_DIR = r"F:\fish\fish\2026\2\LLM机器学习\DeLTa-main\example_datasets"

def download_california_housing():
    """下载 california_housing 数据集"""
    try:
        from sklearn.datasets import fetch_california_housing
        
        print("下载 california_housing 数据集...")
        data = fetch_california_housing()
        
        # 分割数据
        n_train = 16560
        n_val = 2069
        n_test = 2069
        
        X = data.data
        y = data.target
        
        X_train = X[:n_train]
        X_val = X[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        
        y_train = y[:n_train]
        y_val = y[n_train:n_train+n_val]
        y_test = y[n_train+n_val:]
        
        # 保存
        dataset_dir = os.path.join(BASE_DIR, "california_housing")
        np.save(os.path.join(dataset_dir, "N_train.npy"), X_train)
        np.save(os.path.join(dataset_dir, "N_val.npy"), X_val)
        np.save(os.path.join(dataset_dir, "N_test.npy"), X_test)
        np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
        np.save(os.path.join(dataset_dir, "y_val.npy"), y_val)
        np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)
        
        print(f"[OK] california_housing done!")
        print(f"  - train: {X_train.shape}")
        print(f"  - val: {X_val.shape}")
        print(f"  - test: {X_test.shape}")
        
    except Exception as e:
        print(f"✗ california_housing 失败: {e}")

if __name__ == "__main__":
    download_california_housing()
    print("完成！")
