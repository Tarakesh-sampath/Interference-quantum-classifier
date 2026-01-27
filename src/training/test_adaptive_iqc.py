import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.IQL.models.adaptive_iqc import AdaptiveIQC
from src.utils.paths import load_paths


def main():
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    _, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]

    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))  # ±1

    train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # quantum-safe normalization
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Train AdaptiveIQC
    # -------------------------------------------------
    model = AdaptiveIQC(
        K_init=3,
        eta=0.1,
        percentile=5,
        consolidate=True,
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ AdaptiveIQC Test Accuracy:", round(acc, 4))


if __name__ == "__main__":
    main()
