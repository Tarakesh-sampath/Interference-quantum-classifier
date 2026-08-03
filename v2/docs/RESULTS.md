# IQC v2 — Results Summary

All numbers are on the **held-out PCam test split (32,768 patches)** unless noted.
CNNs retrained with the corrected transforms (v1 silently dropped normalization +
augmentation). Multi-seed = 5 seeds; `mean ± std`. Reproduce via `bash v2/run_all.sh`.

> **Headline.** The IQC decision is a classical linear classifier executed on a
> quantum circuit; its only genuine quantum-execution property is measurement-
> efficient *sign* readout. On the honest test split every headline number in the
> paper drops, the paper's noise-resilience claim does not survive a realistic
> circuit depth, and the multi-prototype regimes (absent from the paper) are what
> actually close the gap to kNN.

## 1. Quantumness audit (gate)

`v2/experiments/run_equivalence.py` — see `v2/docs/math_foundations.md`.

- Statevector Hadamard-test vs numpy dot product: `max|Δ| = 1.2×10⁻¹²`.
- 8192-shot circuit vs exact: `max|Δ| = 1.7×10⁻²` (= shot noise `~1/√N`).
- Static IQC ≡ `sign(χ·x)`: **100%** agreement; χ real.
- Fixed-K=3 IQC collapses to **one hyperplane** `W=Σwᵢmᵢ`: **100%** agreement.

## 2. Split hygiene: v1 numbers were inflated

Evaluating on the real PCam test split instead of a 70/30 cut of the first 5000
val patches drops every method substantially:

| method | v1 (val subset) | v2 (test split) |
|---|---|---|
| kNN(5)          | 0.926 | 0.824 ± 0.004 |
| IQC static/K=1  | ~0.876 | 0.802 ± 0.028 |
| IQC fixed K=3   | ~0.886 | 0.806 ± 0.010 |

The ~0.09 kNN gap is the PCam val→test distribution shift, previously hidden.

## 3. Capacity: multi-prototype IQC beats kNN (answers the paper's own limitation)

`v2/results/capacity_sweep/` — IQC accuracy grows monotonically with prototypes and
crosses kNN at **K ≥ 11** (McNemar p < 10⁻²⁵):

| K | 1 | 3 | 5 | 7 | 11 | 17 | 23 |
|---|---|---|---|---|---|---|---|
| acc | 0.802 | 0.806 | 0.815 | 0.818 | 0.835 | 0.838 | 0.839 |

kNN(5) reference = 0.824. The paper only reported K=1/3 and concluded the single-mode
centroid is the bottleneck; v2 confirms this and shows the multi-memory regimes
(Section 4 of the paper's regime ladder) are the fix.

## 4. Adaptive regimes: more accurate with fewer memories

`v2/results/adaptive_regimes/` (5 stream-order seeds):

| model | acc | mean #memories |
|---|---|---|
| Regime-2 perceptron | 0.817 ± 0.029 | 1 |
| Regime-4A/4B adaptive | 0.814 ± 0.009 | 2.8 |
| Regime-3A fixed K=3 | 0.806 ± 0.010 | 6 |

Adaptive beats fixed-K=3 (McNemar p = 4×10⁻⁹) with **less than half the memories**.
Regime-2 has high variance — a single perceptron is sensitive to stream order.

## 5. Shot efficiency scales with margin, not memory count

`v2/results/shot_sweep/` — single-prototype static IQC (median margin 0.338) reaches
its exact-backend accuracy by ~64–128 shots; fixed-K=3 (median margin 0.130) needs
more. Empirical curves lie between the exact binomial prediction and the Hoeffding
bound, converging at high shots. This is the paper's real "measurement efficiency"
result, now derived and margin-quantified.

## 6. Noise resilience does NOT hold at realistic circuit depth

`v2/results/noise_depolarizing*/`, `noise_thermal/`. The exact-encoding ISDO circuit
transpiles to **depth ≈ 1268 (≈1128 1q + ≈462 2q gates)**, dominated by controlled
state-preparation. Depolarizing survival threshold is **p ≈ 1–3×10⁻⁴**:

| p | 1e-5 | 1e-4 | 3e-4 | 1e-3 | 1e-2 |
|---|---|---|---|---|---|
| static IQC acc | 0.824 | 0.824 | 0.797 | 0.508 | 0.499 |

Analytic and `qiskit-aer` circuit tiers agree (both chance for p ≥ 10⁻³). The paper's
Table II claims 88% at **p = 0.01** — ~100× past the threshold; at that noise the
circuit classifies at chance. Thermal: signal lost at T1≈30 µs (typical NISQ),
recovers only near T1≈100–200 µs.

## 7. Main comparison (all baselines, 5 per-seed CNNs)

`v2/results/main_comparison/` — 5 seeds, a freshly retrained CNN per seed, PCam test
split. QSVM/VQC capped at fit=3500/eval=2000 (footnote).

| method | test acc (mean ± std) | notes |
|---|---|---|
| Linear SVM | **0.8405 ± 0.011** | best; tied with LogReg (McNemar p=0.53) |
| Logistic Regression | 0.8404 ± 0.011 | |
| kNN(5) | 0.8336 ± 0.011 | |
| QSVM (fidelity kernel) | 0.8321 ± 0.010 | measurement-based quantum |
| **IQC static (Regime-1)** | 0.8267 ± 0.014 | the paper's proposal |
| IQC fixed K=3 (Regime-3A) | 0.8136 ± 0.020 | |
| IQC adaptive (Regime-4A/B) | 0.8089 ± 0.028 | 6.4 memories avg |
| VQC | 0.7162 ± 0.027 | COBYLA, 1-layer ansatz |

**Reading.** On the honest test split the classical linear methods lead (~0.840); the
single-prototype IQC (0.827) is the best of the IQC family and beats VQC decisively,
but trails LogReg/LinSVM/kNN/QSVM — every pairwise difference except LinSVM–LogReg is
significant (McNemar, 32k paired samples). This is consistent with the equivalence
theorem: the IQC is a linear classifier, so it cannot beat a well-regularized linear
classifier (LinSVM/LogReg) on the same features. Its value proposition is the
measurement-efficient sign readout (§5), not accuracy. The multi-prototype route
(§3, pinned-CNN capacity sweep) is what lets IQC exceed kNN — but only at K≥11, and
was never reported in the paper.

### Bottom line vs the paper

| paper claim | v2 finding |
|---|---|
| 88.07% headline accuracy | inflated by val-subset eval; 0.827 ± 0.014 on true test |
| noise-resilient at p=0.01 | chance-level at p=0.01; survives only to p≈10⁻⁴ (depth ~1268) |
| single-centroid is the bottleneck | confirmed; multi-prototype (K≥11) fixes it, absent from paper |
| "quantum" classification | provably a linear classifier; only sign-readout is quantum-efficient |
