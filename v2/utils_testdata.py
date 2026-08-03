"""Tiny helper: load a few normalized v1 embedding states for unit tests."""

import numpy as np

from src.utils.load_data import load_data


def load_eval_states(n=40):
    """Return (Xte[:n] L2-normalized, a normalized chi probe)."""
    _, Xte, _, _ = load_data("polar")
    Xte = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
    chi = Xte[0]
    return Xte[1 : n + 1], chi
