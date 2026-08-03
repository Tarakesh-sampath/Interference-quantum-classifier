# IQC — Mathematical Foundations (v2)

This document states precisely what the Interference Quantum Classifier (IQC)
computes, proves what it is equivalent to, and derives the one property that is
genuinely a quantum-execution advantage. Empirical checks that confirm each claim
are produced by `v2/experiments/run_equivalence.py` (numbers below are from the v1
embeddings; the audit is re-run on v2 embeddings in the pipeline).

---

## 1. Setup: encoding and the ISDO observable

A frozen CNN maps a patch to an embedding `x ∈ R^d` (`d = 32`, ReLU head, so
`x_i ≥ 0`). Amplitude encoding is

    Φ(x) = |ψ⟩ = x / ‖x‖ ∈ C^{2^n},   n = ⌈log₂ d⌉ = 5.

Because the CNN head is real and non-negative, `|ψ⟩` has **real, non-negative
amplitudes** — no phase is ever populated (audit: `embeddings_real=True`,
`min=0.0`).

A class-representative state `|χ⟩` (a prototype / centroid) is likewise real. The
**Interference-Sign Decision Observable** is

    O(ψ; χ) = Re⟨χ|ψ⟩,     ŷ = sign(O).

The Hadamard test estimates `O` by putting an ancilla in
`(|0⟩+|1⟩)/√2`, applying the controlled transition `U_{χψ}` (`U_ψ` then `U_χ†`),
a final Hadamard, and measuring the ancilla:

    P(0) = ½ + ½ Re⟨χ|ψ⟩,   ⟨Z_anc⟩ = P(0) − P(1) = Re⟨χ|ψ⟩.        (1)

---

## 2. Equivalence theorem — the decision is classical linear algebra

**Claim.** With real encoding, every IQC decision rule in this repo is a linear
classifier in `x`.

**Real-encoding lemma.** For real `χ, ψ`,
`Re⟨χ|ψ⟩ = Σ_i χ_i ψ_i = χ·ψ`. The imaginary machinery of (1) is vestigial: the
observable is the Euclidean inner product. *(Audit: statevector Hadamard-test vs
`np.vdot` agree to `max|Δ| = 1.2×10⁻¹²` — the circuit reproduces the dot product
to machine precision.)*

**(a) Static IQC = signed-centroid classifier.** `StaticISDOClassifier` builds
`χ = normalize(Σ_{c=0} p − Σ_{c=1} p)` from per-class prototypes `p` and predicts
`1 if χ·x < 0 else 0`. This is exactly `sign(−χ·x)` — a single hyperplane whose
normal is the difference of class centroids (nearest-signed-centroid / LDA-with-
identity-covariance). *(Audit: 100% label agreement with the `sign(χ·x)` replica,
`chi_real=True`.)*

**(b) Frozen FixedMemoryIQC = one hyperplane.** After Winner-Take-All training,
inference uses `WeightedVoteClassifier`, which **sums** memory overlaps:

    score(x) = Σ_i w_i (m_i · x) = ( Σ_i w_i m_i ) · x = W · x,   ŷ = sign(W·x).

So the K-prototype model, despite `2K` stored states, collapses at inference to a
**single effective hyperplane** `W = Σ_i w_i m_i`. *(Audit: K=3, 6 memories → 100%
agreement with `sign(W·x)`.)* Winner-Take-All selection during *training* is
nonlinear, but the deployed classifier is not.

**Consequence.** The accuracy ceiling is set by whether the CNN feature map places
the two classes in linearly separable convex regions — not by anything quantum.
This is why plain LogReg on the same embeddings matches or beats the IQC.

---

## 3. Shot-complexity — the real (and only) quantum-execution advantage

The advantage is not in the decision function but in **how cheaply the sign of a
`d`-dimensional overlap can be read out**.

The ancilla is a Bernoulli variable with `P(0) = (1 + s)/2`, `s = χ·x`. With `N`
shots the estimator `ŝ = (2·n₀ − N)/N` is unbiased with the empirical margin
`m = |s|`. By Hoeffding, `P(sign(ŝ) ≠ sign(s)) ≤ exp(−N m² / 2)`, so guaranteeing
sign correctness with confidence `1 − δ` needs

    N ≳ 2 ln(1/δ) / m².                                              (2)

Crucially `N` is **independent of `d`**: the sign of an overlap in a `2^n`-dim space
costs `O(1/m²)` shots, whereas fidelity-based methods estimating `|⟨χ|ψ⟩|²` to
precision `ε` cost `O(1/ε²)` with `ε` needing to resolve the squared quantity.
This is the honest content of the paper's "measurement efficiency" claim.

*Empirical (v2, PCam test split, pinned CNN seed-0).* The single-prototype static
IQC has a median margin `m ≈ 0.338` and its sampled accuracy is already within
Monte-Carlo error of the exact-backend 0.8248 by ~64–128 shots; the multi-memory
fixed-K=3 model has a much smaller median margin `m ≈ 0.130`, so it needs more
shots for the same stability. Per (2), the shot budget scales as `1/m²`, so the
*single* prototype is the more measurement-efficient object — a concrete instance
of the margin, not the memory count, driving shot cost. The empirical shot curves
sit between the exact binomial sign-flip prediction (tight) and the Hoeffding bound
(loose), converging to the exact-backend accuracy at high shots
(`v2/results/shot_sweep/`).

**The asterisk (state preparation).** Amplitude-encoding an arbitrary `x` costs
`O(d)` gates (Plesch–Brukner). This is linear in `d` = exponential in `n`, so the
end-to-end asymptotic advantage over classical `sign(w·x)` (also `O(d)`) is *not*
established; the win is specifically in the measurement/readout budget once states
are prepared, relevant when preparation is amortized or hardware-native.

---

## 4. Noise propagation

**Depolarizing.** A depolarizing channel of strength `p` per layer replaces the
state by the maximally mixed state with probability `p`; over circuit depth `L`
the ancilla expectation attenuates multiplicatively:

    ⟨Z_anc⟩_noisy = (1 − p)^L · Re⟨χ|ψ⟩.                             (3)

The factor is **positive**, so the *sign* — hence the classification — is
preserved; noise only shrinks the margin. Combined with (2), a depolarized margin
`m(1−p)^L` inflates the required shots by

    N_noisy / N ≈ (1 − p)^{−2L}.                                     (4)

`L` is the transpiled depth of the real circuit (logged in the circuit tier of the
noise experiment) and is the same `L` used in the analytic tier — this is what
closes theory (3)–(4) against `qiskit-aer` simulation.

*Empirical (v2) — this is a substantive correction to the paper.* The ISDO circuit
built from exact amplitude encoding transpiles (basis `sx,x,rz,cx`) to
**depth ≈ 1268, with ≈1128 one-qubit and ≈462 two-qubit gates** — the cost is
dominated by the *controlled* arbitrary-state-preparations, not the ancilla. At that
depth the attenuation `(1−p)^{n₁}(1−p₂)^{n₂}` (with 2-qubit rate `p₂ = 10p`) is
severe:

| depolarizing p | attenuation | static IQC acc @1024 shots |
|---|---|---|
| 1e-5 | 0.944 | 0.824 |
| 1e-4 | 0.563 | 0.824 |
| 3e-4 | 0.178 | 0.797 |
| 1e-3 | 0.003 | 0.508 (chance) |
| 1e-2 | ~0    | 0.499 (chance) |

The **survival threshold is `p ≈ 1–3×10⁻⁴`**. The analytic and `qiskit-aer` circuit
tiers agree (both collapse to chance for `p ≥ 10⁻³`), validating the model. The
paper's Table II reports 88% at **p = 0.01**, two orders of magnitude past this
threshold — at that noise level the exact-encoding ISDO circuit classifies at
chance. Equation (12) in the paper is correct but silently assumes a shallow `L`;
for exact amplitude encoding `L ~ 10³`, and the noise-resilience claim does not hold.
Thermal relaxation shows the same picture: signal is lost at `T1 ≈ 30 µs` (typical
NISQ) and only recovers near `T1 ≈ 100–200 µs` (`v2/results/noise_thermal/`).

**Dephasing.** Ancilla dephasing adds a stochastic phase θ:
`P(0) = ½ + ½ Re(e^{iθ}⟨χ|ψ⟩)`. Unbiased dephasing preserves the expected sign;
only a systematic phase drift can move the boundary. The ISDO readout is therefore
more robust to stochastic depolarization than to fixed calibration/phase error.

---

## 5. Honest positioning of the regime ladder

- **Regime 1 (static)** — signed-centroid linear classifier (§2a).
- **Regime 2 (online)** — the update `χ ← normalize(χ + η y ψ)` is the classical
  **perceptron** in `C^{2^n}`; perceptron convergence theory applies.
- **Regimes 3/4 (multi-memory, spawn/prune)** — online **Learning Vector
  Quantization**: prototypes per class, winner updates, growth and pruning. Frozen
  Regime-3A inference still collapses to one hyperplane (§2b); the adaptive
  Regime-4 models are the only ones whose *inference* (if using WTA selection
  rather than the vote) is genuinely piecewise-linear / non-linear.

**Summary.** The IQC is a *quantum-executable linear classifier* with a
measurement-efficient sign readout. That framing is defensible and is what v2
measures; "quantum-enhanced classification" is not supported by the decision
mathematics.
