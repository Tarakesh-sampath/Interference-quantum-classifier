---

# Direction 1 Summary

## Linear Interference–Sign Decision Observable (ISDO)

### Status: **EXPLORED / CLOSED**

---

## 1. Initial Goal of Direction 1

The objective of **Direction 1** was to design a **quantum classification primitive** that:

* uses **linear (signed) similarity**, not quadratic fidelity
* enables **sign-based decision making**
* avoids probabilistic, shot-heavy estimation
* is compatible with **unitary-only quantum mechanics**
* can be physically implemented (at least in principle)

The target observable was fixed as:

$$
\boxed{
\mathcal{O}_{\text{ISDO}}(\psi)
= \operatorname{Re}\langle \chi \mid \psi \rangle
}
$$

where:

* $|\psi\rangle$ is a test embedding
* $|\chi\rangle$ is a class reference superposition

---

## 2. Stage A — Conceptual Definition (Circuit A)

### What was done

* Defined ISDO using a **Hadamard-test-style interference circuit**
* Used an **oracle model** with abstract unitaries:
  * $U_\psi |0\rangle = |\psi\rangle$
  * $U_\chi |0\rangle = |\chi\rangle$
* Constructed a conceptual circuit that interferes $|\psi\rangle$ and $|\chi\rangle$

### Key result

* Circuit A **correctly defines** the observable:
  $$
  \langle Z\rangle = \operatorname{Re}\langle \chi \mid \psi \rangle
  $$

### Key insight

* Circuit A is **definition-only**:
  * pedagogically useful
  * standard in quantum algorithms literature
  * **not physically realizable as-is**

### Status

* ✅ Conceptually correct
* ❌ Not intended for hardware
* ✅ Retained as the **formal definition** of ISDO

---

## 3. Stage B — First Physical Attempt (Reflection-Based Circuit)

### What was attempted

* Implement ISDO physically using:
  * ancilla
  * Hadamard
  * **controlled reflection**
    $$
    R_\chi = I - 2|\chi\rangle\langle\chi|
    $$

### Observed behavior

The circuit consistently measured:

$$
\boxed{
\langle Z\rangle
= 1 - 2|\langle \chi \mid \psi \rangle|^2
}
$$

### Key realization

* This observable is:
  * **quadratic**
  * **phase-insensitive**
  * equivalent to a **fidelity-based classifier**
* It is **not** ISDO.

### Critical insight

> A single controlled reflection + Hadamard test **cannot produce linear overlap**.

This was confirmed analytically and numerically.

### Outcome

* The circuit was **not wrong**, but **measured a different observable**
* This method was **renamed** and separated as a new direction:
  * **RFC — Reflection-Fidelity Classifier**

### Status

* ❌ Does not implement ISDO
* ✅ Retained as a **valid alternative classifier**

---

## 4. Stage C — Disambiguation of Linear vs Quadratic Similarity

Through systematic testing, the following distinction was established:

| Method | Observable | Description |
| :--- | :--- | :--- |
| **ISDO** | $\operatorname{Re}\langle \chi \mid \psi \rangle$ | Linear Interference |
| **RFC** | $1 - 2|\langle \chi \mid \psi \rangle|^2$ | Quadratic Fidelity |

### Empirical observations

* ISDO:
  * preserves sign
  * distinguishes directionality
  * outputs 0 for orthogonal states
* RFC:
  * collapses sign
  * outputs +1 for orthogonal states
  * behaves as a distance-like metric

This confirmed that **ISDO and RFC are fundamentally different classifiers**.

---

## 5. Stage D — Correct Physical Implementation (ISDO-B′)

### Core idea

To physically realize ISDO, the circuit must implement **linear interference**, not a reflection expectation.

This was achieved by introducing a **transition unitary**:

$$
\boxed{
U_{\chi\psi} = U_\chi U_\psi^\dagger
}
\quad\text{such that}\quad
U_{\chi\psi}|\psi\rangle = |\chi\rangle
$$

### Circuit structure (ISDO-B′)

```
Ancilla: |0⟩ ──H──●────H──Z
                   │
Data:    |ψ⟩ ─────Uχψ────
```

### Result

The ancilla measurement yields:

$$
\boxed{
\langle Z\rangle = \operatorname{Re}\langle \chi \mid \psi \rangle
}
$$

### Validation

* Verified numerically against:
  * analytic inner product
  * Circuit A reference
* Tested across:
  * identical states
  * orthogonal states
  * opposite states
  * generic states
* Agreement confirmed to floating-point precision

### Status

* ✅ Correct
* ✅ Unitary
* ✅ Ancilla-based
* ✅ Physically meaningful (oracle-level)
* ✅ **Final ISDO implementation**

---

## 6. Final Architecture for Direction 1

| Component          | Role                             | Status   |
| ------------------ | -------------------------------- | -------- |
| Circuit A          | Conceptual definition            | Complete |
| ISDO-B′            | Physical ISDO implementation     | Complete |
| RFC                | Alternative quadratic classifier | Complete |
| Tests & validation | Correctness proof                | Complete |

---

## 7. Key Conclusions from Direction 1

1. **Linear similarity ≠ quadratic fidelity**
2. ISDO captures **directional, signed interference**
3. Reflection-only methods cannot realize ISDO
4. Transition-based interference is the **minimal correct physical mechanism**
5. Sign-based quantum inference is feasible with **low-shot, ancilla-only measurement**

---

## 8. Final Status Declaration

> **Direction 1 — Linear Interference–Sign Decision Observable (ISDO)**
> **Status: FULLY EXPLORED AND CLOSED**

All conceptual, physical, and numerical questions for this direction have been resolved.

---
