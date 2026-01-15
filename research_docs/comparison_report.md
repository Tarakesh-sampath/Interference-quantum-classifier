# Literature Comparison Report: Measurement-Free Quantum Classifier

This report compares the proposed **Measurement-Free Quantum Classifier** (MFQC) project with the provided literature set (Bucket A, Bucket B, and Scholar Review).

## 1. Technical Innovation: The SWAP-Test Advantage
Most quantum classifiers in current literature (e.g., **Singh 2024** in Bucket A, **VQFE** in Bucket B) rely on:
- **Variational circuits** that require intermediate measurements for gradient estimation (Parameter-Shift Rule).
- **Fidelity estimation** that involves multiple shots to reconstruct matrix elements.

**The MFQC Project** innovates by using a **SWAP-test** protocol. This allows for a **measurement-free** classification process where:
- Quantum coherence is preserved until the final readout bit.
- Only a single ancilla qubit is measured at the very end to determine the fidelity (similarity) between the test image and the class prototypes.
- This directly addresses the research gap identified in **Radhi et al. (2025)**.

## 2. Encoding and Dimension Reduction
Standard quantum image processing papers often struggle with the "curse of dimensionality":
- **Literature (Bucket A)**: Evaluates FRQI/NEQR which require a qubit or gate per pixel, making them impractical for 96x96 medical images.
- **MFQC Approach**: Uses a **Hybrid CNN backbone**. The classical CNN extracts high-level features (16-32D), which are then encoded using **Amplitude Encoding** into only 4-5 qubits. This hybrid approach is supported by **Springer Nature (2023)** as the most viable NISQ-era path.

## 3. NISQ Hardware Feasibility
- **Literature (Scholar Review)**: Highlights that decoherence and noise limit circuit depth to <150 gates for meaningful results.
- **MFQC Approach**: Specifically targets a **shallow circuit design (50-100 gates)**. By avoiding intermediate measurements, it reduces the accumulation of shot noise and readout error, which are major bottlenecks discussed in **MDPI (2024)**.

## 4. Performance against Baselines
| Metric | Literature Average (Standard VQC) | MFQC Proposed Target |
| :--- | :--- | :--- |
| **Circuit Depth** | 150-500+ gates | 50-100 gates |
| **Qubit Count** | High (for raw pixels) | 4-5 (for amplitude features) |
| **Accuracy (Medical)**| 85-90% | **92%** |
| **Coherence** | Interrupted by measurements | **Preserved until final readout** |

## Summary of Gap Filling
The MFQC project sits at the intersection of **Hybrid Machine Learning** (Scholar Review) and **Quantum State Comparison** (Bucket B). It moves beyond the "survey phase" (Radhi 2025) into a practical implementation that leverages the efficiency of the SWAP-test to achieve medical-grade classification without the overhead of measurement-based variational loops.
