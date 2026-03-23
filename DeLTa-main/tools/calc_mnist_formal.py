import numpy as np
from sklearn.metrics import accuracy_score

# Calculate bank traditional_rf accuracy
accs = []
labels = None
for i in range(10):
    d = np.load(f'results/bank/RF_md20_ml50_tree15_full_cart_{i}.npy', allow_pickle=True).item()
    pred = np.asarray(d['logit'])
    label = np.asarray(d['label'])
    # For binary, threshold at 0.5
    if pred.ndim > 1:
        pred = np.argmax(pred, axis=1)
    else:
        pred = (pred > 0.5).astype(int)
    if labels is None:
        labels = label
    accs.append(accuracy_score(label, pred))

print(f"bank traditional_rf mean accuracy: {np.mean(accs):.4f}")
