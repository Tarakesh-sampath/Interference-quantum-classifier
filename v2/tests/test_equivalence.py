"""Equivalence gate as a pytest (mirrors run_equivalence.run).

Run: uv run python -m pytest v2/tests/test_equivalence.py -q
"""

import numpy as np

from v2.experiments.run_equivalence import run


def test_gate_passes_on_v1_embeddings():
    rep = run(n_eval=800)
    b = rep["backends"]
    assert b["max_abs_diff_exact_vs_transition"] < 1e-10, b
    assert b["max_abs_diff_exact_vs_sampled"] < 5e-2, b
    assert rep["static_linear_K1"]["agreement"] == 1.0, rep["static_linear_K1"]
    assert rep["fixed_single_hyperplane_K3"]["agreement"] == 1.0, rep["fixed_single_hyperplane_K3"]
    assert rep["GATE_PASSED"] is True


def test_embeddings_are_real():
    rep = run(n_eval=100)
    assert rep["embeddings_real"] is True
