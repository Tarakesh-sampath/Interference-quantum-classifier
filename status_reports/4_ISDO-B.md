# ISDO-B′ and Hadamard Interference Backends

## Purpose of This Document

This document records the **design rationale, implementation plan, and verification protocol** for introducing **quantum-circuit-based interference backends** into the IQC (Interference Quantum Classifier) framework.

It exists to:
- Preserve architectural intent
- Enable back‑tracing of design decisions
- Support paper, thesis, or patent drafting
- Prevent accidental regression to ad‑hoc circuit integration

This document should be treated as **authoritative design reference** for circuit-backed IQC execution.

---

## Background: What IQC Requires from a Circuit

IQC is defined entirely in terms of a single scalar observable:

> **Interference score**  
> \[ S(\psi, \chi) = \mathrm{Re}\langle \chi \mid \psi \rangle \]

All learning regimes (Regime 2 → 3‑A → 3‑B → 3‑C) depend *only* on this value:
- sign
- relative magnitude
- ordering across samples

IQC **does not** require probabilities, fidelities, or variational gradients.

Therefore, a circuit backend is *only* responsible for **estimating this scalar**, not for learning.

---

## Architectural Principle (Locked)

> **Learning logic must be completely independent of how interference is measured.**

To enforce this, IQC introduces an **Interference Backend abstraction**:

```
score(chi, psi) → float
```

All IQC regimes call this interface. The backend may be:
- mathematical (NumPy)
- circuit‑based (Hadamard test)
- observable‑engineered (ISDO‑B′)

Learning code never inspects or controls the backend.

---

## The Three Interference Backends

### 1. MathInterferenceBackend (Ground Truth)

**Definition**
```
S_math = Re⟨χ|ψ⟩
```

**Purpose**
- Exact reference
- Regression baseline
- Debug oracle

**Properties**
- Deterministic
- Noise‑free
- Not hardware‑executable

This backend defines *semantic correctness* of IQC.

---

### 2. Hadamard-Test Interference Backend (Canonical Quantum Reference)

**Definition**

A Hadamard test with an ancilla qubit yields:

\[ \langle X_{anc} \rangle = \mathrm{Re}\langle \chi | \psi \rangle \]

**Circuit Characteristics**
- Requires controlled state preparation
- One ancilla qubit
- Moderate circuit depth

**Purpose**
- Quantum‑exact realization of the math backend
- Verification oracle for other quantum observables

**Role in Project**
- Reference backend
- Not the final hardware‑efficient solution

This backend answers:
> “Does the circuit produce the same interference value as the math model?”

---

### 3. ISDO‑B′ Interference Backend (Hardware‑Efficient, Novel)

**Conceptual Shift**

ISDO‑B′ does **not** measure ⟨χ|ψ⟩ directly.

Instead, it:
- Embeds χ into an engineered observable
- Applies a fixed interferometric circuit
- Measures a single expectation value

Resulting in:

\[ S_{ISDO}(\psi) = \langle \psi | \mathcal{O}_{\chi} | \psi \rangle \]

where \( \mathcal{O}_{\chi} \) is constructed from χ but **requires no controlled‑χ unitary**.

**Key Properties**
- Shallow circuits
- No controlled state preparation
- NISQ‑friendly
- Observable‑engineered

**Why This Matters**
- Aligns with IQC’s sign‑centric philosophy
- Avoids fidelity / kernel / SWAP paradigms
- Enables strong hardware efficiency claims

This backend is the **intended final embodiment** of IQC.

---

## Verification Philosophy (Critical)

ISDO‑B′ is **not required** to numerically equal the Hadamard test.

Instead, it must satisfy **three sufficiency criteria**:

### Level 1 — Sign Agreement (Required)
```
sign(S_ISDO) == sign(Re⟨χ|ψ⟩)
```

IQC decisions depend on sign.

---

### Level 2 — Ordering / Monotonicity (Required)

For fixed χ and multiple ψ:
```
Re⟨χ|ψ₁⟩ > Re⟨χ|ψ₂⟩  ⇒  S_ISDO(ψ₁) ≥ S_ISDO(ψ₂)
```

This preserves:
- margin logic
- percentile‑based memory growth

---

### Level 3 — Correlation (Optional)

Statistical correlation between:
- Hadamard scores
- ISDO‑B′ scores

Useful for diagnostics, not correctness.

---

## Verification Harness (Planned)

A dedicated script will compare all three backends:

```
MathInterferenceBackend
HadamardInterferenceBackend
ISDOBPrimeInterferenceBackend
```

Across:
- random χ, ψ pairs
- small‑qubit systems
- controlled noise‑free simulation

Metrics recorded:
- sign agreement rate
- ordering violations
- correlation plots

This script is the **scientific validation artifact** for ISDO‑B′.

---

## Integration Test (Final Proof)

After standalone verification:

1. Fix a trained IQC memory bank
2. Swap backends:
   - math → hadamard → ISDO‑B′
3. Run identical IQC inference

Compare:
- predictions
- accuracy
- margin statistics
- Regime‑3C growth triggers

If IQC behavior is preserved, ISDO‑B′ is validated as a backend.

---

## Why This Design Is Strong

### Scientifically
- Separates algorithm from measurement
- Proves sufficiency, not equality
- Matches physical constraints of NISQ hardware

### Architecturally
- No `if use_circuit` pollution
- Backend injection at one point
- Future backends drop‑in

### For IP / Publications
- Multiple independent embodiments
- Observable‑engineered learning
- Circuit‑agnostic algorithm claims

---

## Locked Decisions

- IQC learning code remains circuit‑agnostic
- InterferenceBackend is the only boundary
- Hadamard backend is reference only
- ISDO‑B′ is the target hardware backend
- Verification is based on sign and ordering

---

## Status

**This document reflects the agreed‑upon design as of the current development checkpoint.**

Any deviation from this structure should be treated as an intentional research fork and documented separately.

