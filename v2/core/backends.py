"""Evaluation backends for IQC scoring in v2.

Three modes, all producing an estimate of the real overlap s = Re<chi|psi>:

* ``exact``   — vectorized numpy dot product over the whole test set at once.
* ``sampled`` — analytic Hadamard test: the ancilla is Bernoulli with
  P(0) = (1+s)/2, so drawing n0 ~ Binomial(shots, P0) and returning
  (2*n0 - shots)/shots is *distributionally identical* to sampling the noiseless
  circuit, but runs on 32768 samples x many shot counts in seconds. Depolarizing
  noise composes analytically as s -> (1-p)^L * s.
* ``circuit`` — the real qiskit-aer circuit via HardwareNativeBackend (for
  validation subsets and thermal noise, which has no clean analytic form).

Plus the shot-complexity helpers used by the shot sweep and math doc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def exact_scores(X: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """s_j = Re<chi|psi_j> for real embeddings = X @ chi."""
    return np.real(np.asarray(X)) @ np.real(np.asarray(chi))


def sampled_scores(
    exact_s: np.ndarray,
    shots: int,
    rng: np.random.Generator,
    attenuation: float = 1.0,
) -> np.ndarray:
    """Analytic Hadamard-test sampling of exact overlaps.

    exact_s : true overlaps s = Re<chi|psi> in [-1, 1].
    attenuation : optional (1-p)^L depolarizing factor applied to s before sampling.
    """
    s = np.clip(np.asarray(exact_s, dtype=float) * attenuation, -1.0, 1.0)
    p0 = 0.5 * (1.0 + s)
    n0 = rng.binomial(shots, p0)
    return (2.0 * n0 - shots) / shots


def hoeffding_shots(margin: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """Shots needed for correct sign at confidence 1-delta: N >= 2 ln(1/delta) / m^2."""
    m = np.asarray(margin, dtype=float)
    with np.errstate(divide="ignore"):
        return 2.0 * np.log(1.0 / delta) / (m ** 2)


def sign_flip_prob(margin: np.ndarray, shots: int) -> np.ndarray:
    """Exact binomial probability that the sampled sign is wrong, per sample.

    For true margin m, P0 = (1+m)/2; a wrong sign needs n0 <= shots/2 (for m>0).
    Uses the binomial CDF (via scipy if available, else a normal approximation).
    """
    m = np.abs(np.asarray(margin, dtype=float))
    p0 = 0.5 * (1.0 + m)
    half = shots / 2.0
    try:
        from scipy.stats import binom

        # P(n0 <= floor(half)); ties at exactly half count as 0.5 flip
        k = np.floor(half).astype(int)
        cdf = binom.cdf(k, shots, p0)
        if shots % 2 == 0:
            cdf = cdf - 0.5 * binom.pmf(int(half), shots, p0)
        return cdf
    except ImportError:  # normal approximation
        mu = shots * p0
        var = shots * p0 * (1.0 - p0)
        z = (half - mu) / np.sqrt(np.maximum(var, 1e-12))
        from math import erf

        return 0.5 * (1.0 + np.vectorize(lambda t: erf(t / np.sqrt(2)))(z))
