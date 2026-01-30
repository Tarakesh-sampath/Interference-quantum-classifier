import os
import numpy as np
from src.utils.paths import load_paths
from src.utils.seed import set_seed
from src.utils.label_utils import ensure_polar
from src.utils.label_utils import ensure_binary

def load_data(y="all", limit=None):
    set_seed(42)

    _, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]

    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y_bin = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))
    y_pol = np.load(os.path.join(EMBED_DIR, "val_labels_polar.npy"))

    y_bin = ensure_binary(y_bin)
    y_pol = ensure_polar(y_pol)

    train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

    if limit :
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        train_idx = train_idx[:limit]
        test_idx = test_idx[:limit]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_bin, y_test_bin = y_bin[train_idx], y_bin[test_idx]
    y_train_pol, y_test_pol = y_pol[train_idx], y_pol[test_idx]

    # Defensive normalization
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test  /= np.linalg.norm(X_test, axis=1, keepdims=True)
    
    if y == "binary":
        return X_train, X_test, y_train_bin, y_test_bin
    elif y == "polar":
        return X_train, X_test, y_train_pol, y_test_pol
    elif y == "all":
        return X_train, X_test, y_train_bin, y_test_bin, y_train_pol, y_test_pol
    else:
        raise ValueError("Invalid value for y")