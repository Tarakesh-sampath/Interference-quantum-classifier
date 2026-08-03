"""Single shared data harness for v2.

Reads the per-seed CNN embeddings written by extract_embeddings_v2, applies
explicit L2 normalization, and offers seeded stratified subsampling. Unlike v1's
load_data, there is no hidden set_seed(42) and no implicit split reuse: callers
pass the split name and an RNG.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from v2.core.io import REPO_ROOT


def emb_dir(cnn_seed: int) -> str:
    return os.path.join(REPO_ROOT, "v2", "results", "embeddings", f"cnn_seed{cnn_seed}")


def load_split(cnn_seed: int, split: str, polar: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X L2-normalized float64, y) for one split of one CNN seed.

    y is {0,1} by default, or {-1,+1} when polar=True.
    """
    d = emb_dir(cnn_seed)
    Xp = os.path.join(d, f"{split}_embeddings.npy")
    yp = os.path.join(d, f"{split}_labels.npy")
    assert os.path.exists(Xp), f"missing {Xp}; run extract_embeddings_v2 --seed {cnn_seed}"
    X = np.load(Xp).astype(np.float64)
    y = np.load(yp).astype(np.int64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    assert not np.any(norms == 0), "zero embedding in loaded split"
    X = X / norms
    if polar:
        y = (y * 2 - 1).astype(np.int64)
    return X, y


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: Optional[int], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-balanced subsample of n rows (n=None or n>=len returns all, shuffled)."""
    idx_all = rng.permutation(len(X))
    if n is None or n >= len(X):
        return X[idx_all], y[idx_all]
    classes = np.unique(y)
    per = n // len(classes)
    picked = []
    for c in classes:
        c_idx = idx_all[y[idx_all] == c][:per]
        picked.append(c_idx)
    sel = np.concatenate(picked)
    sel = rng.permutation(sel)  # mix classes
    return X[sel], y[sel]
