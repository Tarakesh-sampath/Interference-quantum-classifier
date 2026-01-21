# Interference Quantum Classifier (IQC)

## Canonical Algorithm Description, Mathematics, Architecture, Results, and Claims

---

## 1. Problem Setting and Motivation

Near-term quantum machine learning (QML) methods are dominated by variational circuits, kernel-based fidelity estimation, and measurement-heavy inference. These approaches suffer from one or more of the following limitations:

- Shot complexity that scales with dataset size
- Loss of phase information due to quadratic observables
- Unstable training dynamics (barren plateaus)
- Tight coupling between learning and hardware execution

The **Interference Quantum Classifier (IQC)** is introduced as an alternative paradigm. IQC separates *learning* from *quantum inference* and uses **linear quantum interference** rather than probability or fidelity as its decision primitive.

---

## 2. High-Level Algorithm Overview

IQC consists of four logically separated stages:

1. Classical feature extraction
2. Quantum state encoding
3. Quantum interference-based inference (ISDO)
4. Classical learning via quantum-state updates

Crucially, **only Stage 3 requires a quantum circuit**. All learning occurs outside the quantum loop.

---

## 3. Stage I — Classical Feature Extraction

Input data (e.g., images) are processed by a classical encoder such as a convolutional neural network (CNN), producing fixed-length feature vectors:

\[ x \in \mathbb{R}^d \]

These vectors are L2-normalized:

\[ \tilde{x} = \frac{x}{\|x\|_2} \]

This normalization ensures valid quantum state preparation.

---

## 4. Stage II — Quantum State Encoding

Each normalized feature vector is mapped to a quantum state:

\[ \tilde{x} \mapsto |\psi(x)\rangle \in \mathcal{H}_{2^n} \]

Encoding can be amplitude-based or otherwise fixed and deterministic. No trainable quantum parameters are introduced at this stage.

The same encoding is used for both:

- Test states \(|\psi\rangle\)
- Class memory states \(|\chi\rangle\)

---

## 5. Stage III — Interference-Sign Decision Observable (ISDO)

### 5.1 Core Observable

The central decision primitive of IQC is the **linear interference observable**:

\[ \boxed{\mathcal{O}_{\text{ISDO}}(\psi; \chi) = \mathrm{Re}\langle \chi | \psi \rangle} \]

The predicted label is obtained by:

\[ \hat{y} = \mathrm{sign}(\mathcal{O}_{\text{ISDO}}) \]

This observable is:

- Linear in the quantum amplitudes
- Sensitive to relative phase
- Sign-preserving (directional)

It fundamentally differs from quadratic fidelity:

\[ |\langle \chi | \psi \rangle|^2 \]

which discards sign and phase information.

---

### 5.2 Physical Realization via Quantum Circuit

To measure \( \mathrm{Re}\langle \chi | \psi \rangle \), IQC employs an ancilla-assisted **interference circuit**.

Conceptually:

1. Prepare an ancilla qubit in \(|0\rangle\)
2. Create a superposition of two paths
3. Coherently interfere \(|\psi\rangle\) and \(|\chi\rangle\)
4. Measure the ancilla in the Z-basis

The expectation value satisfies:

\[ \langle Z_{\text{anc}} \rangle = \mathrm{Re}\langle \chi | \psi \rangle \]

A transition-based implementation (ISDO-B′) realizes this without controlled reflections and avoids quadratic observables.

---

### 5.3 Phase Sensitivity — Illustrative Example

Consider:

\[ |\psi\rangle = |0\rangle \]
\[ |\chi\rangle = \frac{|0\rangle + e^{i\phi}|1\rangle}{\sqrt{2}} \]

Then:

\[ \langle \chi | \psi \rangle = \frac{1}{\sqrt{2}} \]

If instead:

\[ |\psi_\phi\rangle = \frac{|0\rangle + e^{i\phi}|1\rangle}{\sqrt{2}} \]

Then:

\[ \mathrm{Re}\langle \chi | \psi_\phi \rangle = \frac{1 + \cos \phi}{2} \]

Thus, classification depends explicitly on **relative quantum phase**, which cannot be recovered from fidelity alone.

---

## 6. Stage IV — Learning via Quantum State Evolution

### 6.1 Learned Object

IQC does not train circuits. The only learned object is the **class memory state**:

\[ |\chi\rangle \in \mathcal{H}_{2^n} \]

---

### 6.2 Online Update Rule (Quantum Perceptron)

For a labeled sample \((|\psi\rangle, y)\) with \(y \in \{+1, -1\}\):

1. Compute score:
\[ s = \mathrm{Re}\langle \chi | \psi \rangle \]

2. If \( y \cdot s < 0 \), update:

\[ |\chi'\rangle = \frac{|\chi\rangle + \eta y |\psi\rangle}{\| |\chi\rangle + \eta y |\psi\rangle \|} \]

This update:

- Is linear and deterministic
- Requires no gradients
- Preserves interpretability

---

## 7. Learning Regimes

### Regime 1 — Static Prototype Aggregation

\[ |\chi\rangle = \text{normalize}\left( \sum_k |\phi_k^+\rangle - \sum_k |\phi_k^-\rangle \right) \]

---

### Regime 2 — Online / Incremental Learning

Class state updated sequentially using the rule above.

---

### Regime 3 — Multi-State Quantum Memory

Maintain multiple memory states \(\{|\chi^{(j)}\rangle\}\). Inference uses:

\[ \hat{y} = \mathrm{sign}\left( \max_j \mathrm{Re}\langle \chi^{(j)} | \psi \rangle \right) \]

---

## 8. System Architecture

Pipeline:

Input → CNN → Embedding → Normalization → Quantum State
→ ISDO Circuit → Score → Classical Decision

Learning updates modify only the stored class states.

---

## 9. Empirical Findings

- Removing measurement noise alone does not improve accuracy
- Interference-based aggregation improves expressivity
- Inference runtime reduced from minutes (SWAP) to milliseconds
- No variational training required

---

## 10. Claims (Condensed)

### Core Claims

1. A quantum classifier based on linear interference rather than fidelity
2. A sign-based decision rule derived from \(\mathrm{Re}\langle \chi | \psi \rangle\)
3. Separation of learning and quantum inference

### System Claims

- Quantum memory states representing classes
- Read-only quantum inference circuits
- Few-shot and incremental learning without retraining

### Method Claims

- Online state update rules
- Multi-memory ensemble inference
- Phase-sensitive classification

---

## 11. Paper vs Patent Positioning

### Paper Focus

- Mathematical correctness of ISDO
- Empirical validation
- Comparison with fidelity and variational methods

### Patent Focus

- Linear interference as a decision primitive
- State-based learning architecture
- Hardware-agnostic inference design

---

## 12. Conclusion

IQC establishes a new QML paradigm: **learning quantum states classically and querying them via physical quantum interference**. This design avoids the central limitations of NISQ-era quantum classifiers while preserving uniquely quantum decision semantics.

