"""Statistics for v2: cross-seed aggregation and paired McNemar tests."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def aggregate(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(len(a)),
    }


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Paired McNemar test on two classifiers over the same examples.

    b = #(A correct, B wrong), c = #(A wrong, B correct). Exact binomial when the
    discordant count b+c < 25, else chi-square with continuity correction.
    """
    a_correct = np.asarray(pred_a) == np.asarray(y_true)
    b_correct = np.asarray(pred_b) == np.asarray(y_true)
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "method": "identical"}
    try:
        from scipy.stats import binom, chi2

        if n < 25:
            k = min(b, c)
            p = 2.0 * binom.cdf(k, n, 0.5)
            p = min(p, 1.0)
            return {"b": b, "c": c, "statistic": float(k), "p_value": float(p), "method": "exact"}
        stat = (abs(b - c) - 1) ** 2 / n
        p = float(chi2.sf(stat, df=1))
        return {"b": b, "c": c, "statistic": float(stat), "p_value": p, "method": "chi2_cc"}
    except ImportError:
        stat = (abs(b - c) - 1) ** 2 / n
        return {"b": b, "c": c, "statistic": float(stat), "p_value": float("nan"), "method": "chi2_cc_noscipy"}


def mcnemar_matrix(preds: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Pairwise McNemar p-values for a set of named prediction arrays."""
    names = list(preds)
    out: Dict[str, Dict[str, float]] = {n: {} for n in names}
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            r = mcnemar(preds[a], preds[bname], y_true)
            out[a][bname] = r["p_value"]
            out[bname][a] = r["p_value"]
    return out
