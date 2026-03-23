from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

# Test different depths for bank
print("=== bank ===")
X = np.load('example_datasets/bank/N_test.npy')
y = np.load('example_datasets/bank/y_test.npy')
for depth in [3, 4, 5]:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X[:1000], y[:1000])
    acc = rf.score(X, y)
    print(f"depth={depth}: {acc:.4f}")

# Test different depths for mnist
print("\n=== mnist ===")
X = np.load('example_datasets/mnist/N_test.npy')
y = np.load('example_datasets/mnist/y_test.npy')
for depth in [3, 4, 5]:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X[:1000], y[:1000])
    acc = rf.score(X, y)
    print(f"depth={depth}: {acc:.4f}")

# Test different depths for house_16H_reg (regression)
print("\n=== house_16H_reg ===")
X = np.load('example_datasets/house_16H_reg/N_test.npy')
y = np.load('example_datasets/house_16H_reg/y_test.npy')
for depth in [3, 4, 5]:
    rf = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X[:1000], y[:1000])
    pred = rf.predict(X)
    rmse = np.sqrt(np.mean((y - pred)**2))
    print(f"depth={depth}: RMSE={rmse:.2f}")
