"""Deterministic sub-RNG derivation for v2 experiments.

One integer seed per (experiment, seed-index) run; every source of randomness
draws from an independent child stream derived via ``SeedSequence.spawn`` so that
e.g. changing the shot count never perturbs the KMeans prototypes.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable

import numpy as np

DEFAULT_STREAMS = ("subsample", "kmeans", "stream", "shots", "sim")


def make_rngs(seed: int, names: Iterable[str] = DEFAULT_STREAMS) -> Dict[str, np.random.Generator]:
    """Return one independent ``np.random.Generator`` per named stream.

    The children are spawned from a single ``SeedSequence(seed)`` so runs are
    reproducible from the top-level integer alone, and the streams are mutually
    independent (no cross-talk between subsampling, clustering, shot noise, ...).
    """
    names = list(names)
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


def set_global_seed(seed: int) -> None:
    """Seed the process-global RNGs (python ``random``, numpy legacy, torch).

    Use for things that read the global state (torch model init, sklearn's
    ``random_state`` fallbacks). Prefer an explicit generator from
    :func:`make_rngs` wherever the API accepts one.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def spawn_int(rng: np.random.Generator, high: int = 2**31 - 1) -> int:
    """Draw a plain python int seed (e.g. for ``seed_simulator``) from a Generator."""
    return int(rng.integers(0, high))
