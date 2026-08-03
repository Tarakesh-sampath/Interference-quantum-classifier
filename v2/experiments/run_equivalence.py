"""Quantumness audit / equivalence gate.

Establishes empirically what the math_foundations.md derives:

1. Backend equivalence — the "quantum" circuits (statevector TransitionBackend,
   shot-based HardwareNativeBackend) compute exactly Re<chi|psi> = the classical
   ExactBackend dot product. Nothing supra-classical is being accessed; the
   simulation merely reads out a real linear overlap.

2. Static IQC == signed-centroid linear classifier: StaticISDOClassifier's labels
   equal sign(chi . x) for the same chi = normalize(sum protos_0 - sum protos_1).

3. Frozen FixedMemoryIQC == a single effective hyperplane: because
   WeightedVoteClassifier *sums* memory scores,
   sign(sum_i w_i (m_i . x)) = sign((sum_i w_i m_i) . x). The K-prototype model
   collapses to one linear boundary W = sum_i w_i m_i at inference.

Run as a gate on the v1 embeddings before any v2 experiment:

    uv run python -m v2.experiments.run_equivalence

Exit code 0 iff all agreements are exact (labels 100%, |dscore| < tol).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.transition import TransitionBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend
from src.IQL.models.static_isdo_model import StaticISDOModel
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.utils.load_data import load_data

TOL_EXACT = 1e-10          # statevector circuit vs numpy dot
TOL_SAMPLED = 5e-2         # high-shot sampling vs exact (shot noise)


def _l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def audit_backends(chi, psis, high_shots=8192, n_circuit=64, seed=0):
    """Compare Re<chi|psi> across exact / statevector-circuit / shot-circuit."""
    exact = ExactBackend()
    trans = TransitionBackend()
    hw = HardwareNativeBackend(shots=high_shots, seed_simulator=seed)

    s_exact = np.array([exact.score(chi, p) for p in psis])
    s_trans = np.array([trans.score(chi, p) for p in psis[:n_circuit]])
    s_hw = hw.score_batch(chi, psis[:n_circuit])

    d_trans = float(np.max(np.abs(s_exact[:n_circuit] - s_trans)))
    d_hw = float(np.max(np.abs(s_exact[:n_circuit] - s_hw)))
    return {
        "max_abs_diff_exact_vs_transition": d_trans,
        "max_abs_diff_exact_vs_sampled": d_hw,
        "high_shots": high_shots,
        "n_circuit": n_circuit,
        "transition_matches_exact": d_trans < TOL_EXACT,
        "sampled_matches_exact_within_shotnoise": d_hw < TOL_SAMPLED,
    }


def audit_static_linear(Xtr, ytr, Xte, seed=42, K=1):
    """StaticISDOClassifier labels == sign(chi . x) replica."""
    model = StaticISDOModel(K=K, seed=seed).fit(Xtr, ytr)
    chi = model.classifier.chi
    labels_model = model.predict(Xte)
    # Replica: StaticISDOClassifier.predict_one returns 1 if score<0 else 0
    scores = Xte @ np.real(chi)
    labels_replica = np.where(scores < 0, 1, 0)
    agree = float(np.mean(labels_model == labels_replica))
    return {
        "K": K,
        "agreement": agree,
        "is_linear": agree == 1.0,
        "chi_is_real": bool(np.allclose(np.imag(chi), 0)),
    }


def audit_fixed_single_hyperplane(Xtr, ytr, Xte, seed=42, K=3):
    """Frozen FixedMemoryIQC == single hyperplane W = sum_i w_i m_i."""
    model = FixedMemoryIQC(K=K, eta=0.1, backend=ExactBackend(), seed=seed).fit(Xtr, ytr)
    clf = model.classifier
    mem = clf.memory_bank.class_states
    W = np.zeros_like(np.real(mem[0].vector))
    for w, cs in zip(clf.weights, mem):
        W = W + w * np.real(cs.vector)
    preds_model = np.array(model.predict(Xte))
    preds_replica = np.where(Xte @ W >= 0, 1, -1)
    agree = float(np.mean(preds_model == preds_replica))
    return {
        "K": K,
        "n_memories": len(mem),
        "agreement": agree,
        "collapses_to_single_hyperplane": agree == 1.0,
    }


def run(use_v1=True, n_eval=1500):
    Xtr, Xte, ytr, yte = load_data("polar")  # y in {-1,+1}
    Xtr, Xte = _l2(Xtr), _l2(Xte)
    ytr_bin = (ytr > 0).astype(int)

    Xte_eval = Xte[:n_eval]
    chi_probe = Xtr[0]

    report = {
        "source": "v1_embeddings" if use_v1 else "v2",
        "n_train": int(len(Xtr)),
        "n_eval": int(len(Xte_eval)),
        "embeddings_real": bool(np.isrealobj(Xtr)),
        "embeddings_min": float(Xtr.min()),
        "backends": audit_backends(chi_probe, Xte_eval),
        "static_linear_K1": audit_static_linear(Xtr, ytr_bin, Xte_eval, K=1),
        "fixed_single_hyperplane_K3": audit_fixed_single_hyperplane(Xtr, ytr, Xte_eval, K=3),
    }

    passed = (
        report["backends"]["transition_matches_exact"]
        and report["backends"]["sampled_matches_exact_within_shotnoise"]
        and report["static_linear_K1"]["is_linear"]
        and report["fixed_single_hyperplane_K3"]["collapses_to_single_hyperplane"]
    )
    report["GATE_PASSED"] = bool(passed)
    return report


def main():
    rep = run()
    b = rep["backends"]
    print("=== Equivalence / Quantumness Audit (v1 embeddings) ===")
    print(f"embeddings real={rep['embeddings_real']} min={rep['embeddings_min']:.4f}")
    print(f"[backend] exact vs statevector-circuit  max|d| = {b['max_abs_diff_exact_vs_transition']:.2e}  -> {b['transition_matches_exact']}")
    print(f"[backend] exact vs {b['high_shots']}-shot circuit  max|d| = {b['max_abs_diff_exact_vs_sampled']:.2e}  -> {b['sampled_matches_exact_within_shotnoise']}")
    s = rep["static_linear_K1"]
    print(f"[static]  StaticISDO == sign(chi.x) linear: agreement={s['agreement']:.4f} chi_real={s['chi_is_real']}")
    f = rep["fixed_single_hyperplane_K3"]
    print(f"[fixed]   FixedMemoryIQC K=3 ({f['n_memories']} mem) collapses to ONE hyperplane: agreement={f['agreement']:.4f}")
    print(f"\nGATE_PASSED = {rep['GATE_PASSED']}")

    # persist
    from v2.core.io import REPO_ROOT
    import json
    out = os.path.join(REPO_ROOT, "v2", "results", "equivalence")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "audit_v1.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"wrote {os.path.join(out, 'audit_v1.json')}")
    sys.exit(0 if rep["GATE_PASSED"] else 1)


if __name__ == "__main__":
    main()
