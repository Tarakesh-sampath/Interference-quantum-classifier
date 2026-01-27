import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.IQL.models.static_isdo_model import StaticISDOModel

def main():
    # -------------------------------------------------
    # Load paths
    # -------------------------------------------------
    _, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]

    # -------------------------------------------------
    # Load embeddings and labels
    # -------------------------------------------------
    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))  # {0,1}

    train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # -------------------------------------------------
    # Sanity: ensure quantum-safe normalization
    # -------------------------------------------------
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Run Static ISDO Model
    # -------------------------------------------------
    K = 3  # best K from sweep
    model = StaticISDOModel(K=K)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ StaticISDOModel | K={K} | Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
