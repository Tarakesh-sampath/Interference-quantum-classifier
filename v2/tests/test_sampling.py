"""The analytic binomial sampling path must match the noiseless aer circuit.

Both estimate Re<chi|psi> as a Bernoulli ancilla with P(0)=(1+s)/2. On the same
inputs at the same shot count their sampled scores must share the same mean and
variance (up to Monte-Carlo error), confirming the fast path is a faithful stand-in
for circuit execution.

Run: uv run python -m pytest v2/tests/test_sampling.py -q
"""

import numpy as np

from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend
from v2.core.backends import sampled_scores
from v2.utils_testdata import load_eval_states


def test_fast_path_matches_circuit_distribution():
    Xte, chi = load_eval_states(n=40)
    exact = ExactBackend()
    s_true = np.array([exact.score(chi, p) for p in Xte])

    shots = 512
    reps = 40
    rng = np.random.default_rng(0)

    # analytic fast path: many reps -> per-sample empirical mean/var
    fast = np.stack([sampled_scores(s_true, shots, rng) for _ in range(reps)])
    fast_mean, fast_var = fast.mean(0), fast.var(0)

    # real noiseless circuit
    circ = []
    for r in range(reps):
        hw = HardwareNativeBackend(shots=shots, seed_simulator=1000 + r)
        circ.append(hw.score_batch(chi, Xte))
    circ = np.stack(circ)
    circ_mean, circ_var = circ.mean(0), circ.var(0)

    # means track the true overlap; both unbiased
    assert np.max(np.abs(fast_mean - s_true)) < 0.06
    assert np.max(np.abs(circ_mean - s_true)) < 0.06
    # variances agree with the binomial prediction (1-s^2)/shots
    pred_var = (1 - s_true ** 2) / shots
    assert np.max(np.abs(fast_var - pred_var)) < 0.02
    assert np.max(np.abs(circ_var - pred_var)) < 0.02
