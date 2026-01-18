import os
import numpy as np
from sklearn.cluster import KMeans

from src.utils.paths import load_paths
from src.utils.seed import set_seed


# ----------------------------
# Reproducibility
# ----------------------------
set_seed(42)

# ----------------------------
# Load paths
# ----------------------------
_, PATHS = load_paths()
EMBED_DIR = PATHS["embeddings"]
CLASS_DIR = PATHS["class_prototypes"]
os.makedirs(CLASS_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

K = int(PATHS["class_count"]["K"])  # prototypes per class

# ----------------------------
# Load embeddings
# ----------------------------
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))

X_train = X[train_idx]
y_train = y[train_idx]

print("Loaded embeddings:", X_train.shape)

# ----------------------------
# Helper: quantum-safe normalize
# ----------------------------
def to_quantum_state(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x / np.sqrt(np.sum(x ** 2))
    assert np.isclose(np.sum(x ** 2), 1.0, atol=1e-12)
    return x


# ----------------------------
# Compute prototypes per class
# ----------------------------
for cls in [0, 1]:
    X_cls = X_train[y_train == cls].astype(np.float64)

    print(f"\nClustering class {cls} with {len(X_cls)} samples")

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(X_cls)

    centers = kmeans.cluster_centers_

    for i in range(K):
        proto = to_quantum_state(centers[i])
        path = os.path.join(CLASS_DIR, f"class{cls}_proto{i}.npy")
        np.save(path, proto)
        print(f"Saved {path}")
