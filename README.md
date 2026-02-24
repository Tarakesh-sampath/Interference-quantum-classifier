# Interference Quantum Classifier (IQC)

Measurement-free, coherence-preserving hybrid quantum–classical classifier for medical image patches using linear quantum interference.

## Overview

The **Interference Quantum Classifier (IQC)** is a novel paradigm for Quantum Machine Learning (QML) that addresses the limitations of near-term quantum devices: high measurement overhead, statistical noise, and variational training complexity. 

Unlike traditional measurement-based quantum classifiers (like those using SWAP-test fidelity estimation), IQC leverages **linear quantum interference** as its decision primitive. It separates the learning process from quantum inference, allowing class representations to be updated classically while utilizing quantum interference for efficient similarity-based classification.

### Key Features
- **Measurement-Free Inference:** Uses the Interference-Sign Decision Observable (ISDO) to make decisions without repeated circuit executions.
- **Coherence Preservation:** Maintains relative phase information that is typically lost in quadratic fidelity measurements.
- **High Efficiency:** Orders-of-magnitude faster inference compared to measurement-based SWAP-test methods.
- **No Variational Training:** Eliminates barren plateaus and training instability by using deterministic state update rules.

---

## Core Concept: ISDO

The central decision primitive of IQC is the **Interference-Sign Decision Observable (ISDO)**:

$$\mathcal{O}_{\text{ISDO}}(\psi; \chi) = \text{Re}\langle \chi | \psi \rangle$$

The predicted label $\hat{y}$ is obtained by the sign of the interference:
$$\hat{y} = \text{sign}(\text{Re}\langle \chi | \psi \rangle)$$

### Why ISDO?
- **Linearity:** Linear in quantum amplitudes, making it more expressive than quadratic fidelity ($|\langle \chi | \psi \rangle|^2$).
- **Phase Sensitivity:** Classification explicitly depends on the relative quantum phase between the test state $|\psi\rangle$ and the class memory state $|\chi\rangle$.

---

## System Architecture

The pipeline follows a modular hybrid approach:

```mermaid
graph LR
    A[Medical Images] --> B[CNN Encoder]
    B --> C[L2-Normalized Embeddings]
    C --> D[Quantum State Encoding]
    D --> E[ISDO Circuit]
    E --> F[Class Decision]
    F --> G[Binary Prediction]
```

1. **Feature Extraction:** A CNN processes histopathology patches (PatchCamelyon dataset) into 32D features.
2. **State Preparation:** Features are mapped to quantum states using amplitude encoding.
3. **Inference:** The ISDO circuit measures the real part of the inner product between the input and the learned class prototype.
4. **Learning:** Class memory states are updated classically via an online update rule (Quantum Perceptron).

---

## Experimental Results

Evaluated on 5,000 validation samples from the **PatchCamelyon (PCam)** dataset:

| Method | Accuracy | Runtime |
| :--- | :--- | :--- |
| Logistic Regression | 0.909 | Fast |
| k-Nearest Neighbors (k=5) | 0.926 | Fast |
| SWAP Test (1024 shots) | 0.875 | Minutes |
| **IQC (Single Prototype)** | **0.876** | **Milliseconds** |
| **IQC (Multiple Prototypes, K=3)** | **0.886** | **Milliseconds** |

### Key Insight
Experimental results indicate that **representation strategy** (aggregation via interference), rather than measurement precision, is the primary performance bottleneck. IQC achieves comparable or superior accuracy to measurement-based methods while being drastically faster.

---

## Installation & Usage

This project uses `uv` for dependency management.

### Setup
```bash
# Clone the repository
git clone https://github.com/Tarakesh-sampath/measurement-free-quantum-classifier.git
cd measurement-free-quantum-classifier

# Install dependencies and create venv
uv sync
```

### Running Experiments
- **Feature Extraction:** `uv run src/training/classical/extract_embeddings.py`
- **IQC Evaluation:** `uv run src/evaluate_iqc_vs_classical.py`
- **Capacity Sweep:** `uv run src/evaluate_capacity_sweep_quantum_vs_knn.py`

---

## Project Structure

- `src/IQL/`: Core Intervention Quantum Learning library.
  - `backends/`: Quantum simulation backends (Statevector, Circuit).
  - `inference/`: Implementation of ISDO-based decision rules.
  - `learning/`: Update rules for class memory states.
  - `models/`: IQC model definitions.
- `status_reports/`: Detailed research reports and technical documentation.
- `results/`: Cached embeddings and experimental metrics.

---

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
