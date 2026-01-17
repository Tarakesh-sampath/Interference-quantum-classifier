# Comparative Analysis: Fidelity-Based & Measurement-Free Quantum Classification

This report provides a formal technical review of quantum classification methods found in the project's literature repository (`Documents/`). It emphasizes **Measurement-Free (MF)** architectures and **Low-Shot** similarity algorithms.

---

## 1. Coherent Feedback Learning (The Absolute MF Baseline)
**Reference**: [Alvarez-Rodriguez et al. (2017)](file:///home/tarakesh/Work/Repo/measurement-free-quantum-classifier/Documents/refference_papers/Scholar%20review/quantum%20fidelity/s41598-017-13378-0.pdf)

*   **Mechanism**: Encodes the classification logic into a time-delayed Schrödinger equation.
*   **Shot Efficiency**: **Zero mid-circuit shots**. The system evolves unitarily toward the correct label.
*   **Equation**: 
    $$\frac{d}{dt} |\psi(t)\rangle = -i \left[ \kappa_1 H_int + \kappa_2 H_{feedback} \right] |\psi(t)\rangle$$

---

## 2. Coherent Amplitude/Phase Estimation (The Bit-by-Bit Approach)
**Reference**: [Patrick Rall (2021)](file:///home/tarakesh/Work/Repo/measurement-free-quantum-classifier/Documents/refference_papers/Scholar%20review/minimm%20measurement%20quant%20algo/q-2021-10-19-566.pdf)

*   **Mechanism**: Uses **Singular Value Transformation (SVT)** to estimate similarity one bit at a time.
*   **Shot Efficiency**: Achieves **Heisenberg-limited** accuracy ($\Theta(1/\epsilon)$ queries). 
*   **Advantage**: Does not require the Quantum Fourier Transform (QFT), making it much more robust for NISQ devices.
*   **Expression**: 
    $$|0\rangle |\psi\rangle \to |\text{overlap}\rangle |\psi\rangle$$
    This "writes" the fidelity into a register without collapsing the original superposition.

---

## 3. Classical Shadows (Shadow Classification)
**Reference**: [Huang et al. (2020) & Yunfei Wang (2024)](file:///home/tarakesh/Work/Repo/measurement-free-quantum-classifier/Documents/refference_papers/Scholar%20review/NISQ%20hardwere/2401.11351v2.pdf)

*   **Mechanism**: Performs randomized Pauli measurements to create a "shadow" of the quantum state.
*   **Shot Efficiency**: Allows tracking **logarithmic** shots relative to the number of samples. Once a shadow is created, you can compute INFINITE fidelities classically.
*   **Equation**: 
    $$\hat{\rho} = \mathbb{E}[ \mathcal{M}^{-1}(U^\dagger |b\rangle\langle b| U) ]$$
    Where $\hat{\rho}$ is the reconstructed "shadow" that contains the fidelity information.

---

## 4. Destructive SWAP-Test (Ancilla-Free)
**Reference**: [Garcia-Escartin (2013) & Blank (2020)](file:///home/tarakesh/Work/Repo/measurement-free-quantum-classifier/Documents/refference_papers/Scholar%20review/quantum%20fidelity/s41534-020-0272-6.pdf)

*   **Mechanism**: Removes the ancilla qubit entirely. Uses CNOTs followed by single-qubit measurements on both registers.
*   **Shot Efficiency**: Far more efficient for hardware with limited connectivity. 
*   **Equation**: 
    Considers the parity of the measurement outcomes $b_1, b_2$:
    $$F = 1 - 2 \cdot P(\text{parity yields odd})$$

---

## 5. Comparative Shot-Efficiency Table

| Method | Shots Required | Measurement-Free? | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Standard SWAP** | $O(1/\epsilon^2)$ | No | General Purpose |
| **Coherent SVT** | $\Theta(1/\epsilon)$ | **Yes** | High Precision / Coherent Chains |
| **Classical Shadows** | $\log(M)$ | Partial | Multi-class (Benign, Malignant, Cyst) |
| **Destructive SWAP** | Medium | No | Low-Qubit Count Chips |
| **VQFE** | High (Training) | No | Parameter Tuning |

---
### 💡 Project Conclusion
While the "Big 3" get most of the attention in textbooks, recent 2021-2024 research (like **Patrick Rall's SVT**) proves that we can achieve **classification without measurement collapse**. In our project, we use the **Interference Average (Phase B)** as a bridge: it uses the parallel nature of the SWAP-test to reduce the "effective" shots compared to testing prototypes one-by-one.
