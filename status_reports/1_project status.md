# Measurement-Free Quantum Similarity Classification on Medical Image Embeddings

**Author:** Tarakeshwaran  
**Date:** *17 jan 2026*  

---

## Abstract

Quantum machine learning methods are often proposed as replacements for classical classifier heads in deep learning pipelines, but in practice their evaluation is complicated by measurement noise, limited qubit counts, and high execution overhead on near-term quantum hardware. This project investigates whether **measurement-free quantum similarity methods** can serve as an efficient and interpretable alternative to traditional measurement-based quantum classifiers when applied to real medical imaging data.

A convolutional neural network (CNN) is first used as a fixed feature extractor to generate compact embeddings from histopathology image patches drawn from the PatchCamelyon (PCam) dataset, a standard binary medical classification benchmark. Instead of training end-to-end quantum models, the study focuses on **quantum similarity-based classification**, where test embeddings are compared against class-representative quantum states.

Three categories of classifiers are evaluated on the same embeddings: classical baselines (logistic regression, linear SVM, and k-nearest neighbors), a **measurement-based quantum classifier using SWAP-test fidelity estimation**, and a set of **measurement-free quantum classifiers**. The measurement-free approaches include (i) an exact statevector-based fidelity classifier using a single class prototype and (ii) a novel **interference-based quantum similarity classifier** that represents each class as a coherent superposition of multiple prototype states. This design increases representational capacity while avoiding repeated circuit executions and shot-based measurement noise.

Experimental results on 5,000 validation samples show that measurement-based SWAP-test classifiers incur significant runtime overhead while achieving accuracy comparable to exact fidelity evaluation, indicating that measurement noise is not the dominant limiting factor. In contrast, the interference-based measurement-free classifier demonstrates improved accuracy over single-prototype quantum similarity while maintaining orders-of-magnitude faster inference. These findings suggest that **representation and aggregation strategy**, rather than measurement precision, is the primary bottleneck for practical quantum similarity classifiers on near-term devices.

Overall, the project provides a systematic comparison of classical, measurement-based quantum, and measurement-free quantum similarity methods on realistic medical image embeddings, highlighting how quantum interference can be leveraged to improve expressiveness without incurring the costs of repeated measurement.

---

## 1. Problem Statement and Motivation

### 1.1 Problem Statement

Quantum machine learning (QML) is often presented as a promising extension to classical deep learning, particularly for classification tasks where quantum circuits are proposed as replacements for conventional neural network classifier heads. However, in practice, most proposed quantum classifiers rely heavily on **repeated measurements**, variational training, or hardware-dependent assumptions that make it difficult to evaluate their real usefulness on near-term quantum devices.

In parallel, medical image analysis remains a domain where **robust, interpretable, and efficient classifiers** are critical. While deep convolutional neural networks achieve strong performance, their final classification layers are typically classical and opaque. This raises a natural question:

> *Can quantum methods provide a meaningful alternative for classification when applied to real medical image data, without incurring prohibitive measurement overhead or hardware noise sensitivity?*

More specifically, the project addresses the following problem:

> **How do measurement-based and measurement-free quantum similarity classifiers behave when applied to real-world medical image embeddings, and what factors actually limit their performance?**

---

### 1.2 Motivation

The motivation for this project arises from three converging observations:

#### Measurement Overhead in Quantum Classifiers
Most near-term quantum classifiers, including variational quantum classifiers (VQCs) and SWAP-test–based similarity methods, rely on **repeated circuit executions** to estimate probabilities or fidelities. This introduces significant runtime overhead, statistical noise due to finite sampling, and scalability concerns as dataset size increases.

#### Representation vs. Measurement Uncertainty
It is often assumed that **measurement noise** is the dominant source of error in quantum classifiers. This project explicitly tests that assumption by separating measurement effects from representational limitations.

#### Practical Evaluation on Medical Data
Medical imaging provides a realistic testbed for quantum ML ideas. By operating on CNN-generated embeddings rather than raw images, the project isolates the classification problem itself and enables fair comparison between classical and quantum similarity-based decision rules.

---

## 2. Initial Idea and Evolution of the Concept

The project initially aimed to replace a classical CNN classifier head with a quantum classifier. Early designs considered variational quantum classifiers, but these were abandoned due to training complexity, instability, and interpretability concerns.

The project then shifted toward **similarity-based classification**, motivated by the natural alignment between quantum fidelity and similarity measures. This led to the implementation of a measurement-based SWAP-test classifier as an initial quantum baseline.

Subsequent experiments revealed that removing measurement noise via exact statevector evaluation did **not** significantly improve accuracy, indicating that measurement noise was not the primary bottleneck. This insight redirected focus toward **representational capacity**.

The final conceptual shift introduced a **measurement-free, interference-based quantum classifier**, which uses coherent superposition of multiple class prototypes to aggregate similarity at the amplitude level.

---

## 3. System Architecture

The system follows a modular architecture:

```mermaid
    A[Medical Images] --> B[CNN Feature Extractor]
    B --> C[32-Dimensional Embeddings]
    C --> D[Classifier Head Classical or Quantum]
    D --> E[Binary Prediction]
```



The CNN is treated as a fixed embedding generator. All classical and quantum classifiers operate on the same embeddings to ensure fair comparison.

---

## 4. Technology Stack

- **Python 3**
- **PyTorch / Torchvision** – CNN and dataset handling
- **scikit-learn** – classical classifiers and k-means clustering
- **Qiskit + Qiskit Aer** – quantum simulation
- **NumPy** – numerical representation (explicit FP64 usage)
- **Matplotlib** – visualization
- **uv** – environment and dependency management

---

## 5. Dataset and Input Pipeline

- **Dataset:** PatchCamelyon (PCam)
- **Task:** Binary classification (benign vs malignant)
- **Input:** Histopathology image patches
- **Evaluation size:** 5,000 validation samples

Images are preprocessed using standard torchvision transforms. CNN embeddings are extracted once and reused across all experiments.

---

## 6. Models and Algorithms

### Classical Baselines
- Logistic Regression: $\approx 0.909$ accuracy
- Linear SVM: $\approx 0.912$ accuracy
- k-NN: $k = 5, \approx 0.926$ accuracy

### Quantum Methods

#### Measurement-Based SWAP Test
- Fidelity estimated via repeated measurements
- Accuracy $\approx 0.875$
- Runtime: minutes for full evaluation

#### Measurement-Free Single-Prototype (Phase A)
- Exact statevector fidelity
- Accuracy $\approx 0.876$
- Demonstrates measurement noise is not dominant

#### Interference-Based Measurement-Free Classifier (Phase B)
- Multiple prototypes per class (K = 3)
- Coherent superposition and interference
- Accuracy $\approx 0.886$
- Runtime: milliseconds

---

## 7. Iterative Development Log

- Early attempts with VQCs were abandoned
- Extensive debugging of:
  - quantum state normalization
  - FP64 precision issues
  - tensor shape mismatches
- Measurement bottleneck identified and disproven
- Representation bottleneck identified and addressed via interference

---

## 8. Experiments and Results

| Method | Accuracy | Runtime |
|------|---------|--------|
| Logistic Regression | 0.909 | Fast |
| Linear SVM | 0.912 | Fast |
| k-NN (k=5) | 0.926 | Fast |
| SWAP Test (1024 shots) | 0.875 | Minutes |
| Measurement-Free (Phase A) | 0.876 | ms |
| Phase B (K=3) | **0.886** | ms |

---

## 9. Final System Behavior

- Deterministic inference for measurement-free methods
- Significant runtime reduction compared to SWAP test
- Improved accuracy via interference-based aggregation

---

## 10. Limitations

- Simulator-only quantum execution
- Global (non-adaptive) prototypes
- CNN not optimized for maximum embedding separability

---

## 11. Future Improvements

- Increase prototype count (K = 5, 8)
- Adaptive or weighted prototypes
- Circuit-level implementation of Phase B
- Evaluation on additional medical datasets

---

## 12. Reflection

This project demonstrated that:
- measurement noise is not always the main limitation,
- representational capacity is critical,
- quantum interference can improve similarity aggregation,
- careful debugging and iteration are essential in quantum ML research.

---

## 13. Appendix

### Key Commands

```bash
uv run train_cnn.py
uv run extract_embeddings.py
uv run train_embedding_models.py
uv run evaluate_swap_test_batch.py
uv run evaluate_statevector_similarity.py
uv run compute_class_prototypes_k4.py
uv run evaluate_interference_k4.py
```

### Directory Structure

```text
measurement-free-quantum-classifier/
├── src/
├── scripts/
├── results/
│   └── embeddings/
├── configs/
├── pyproject.toml
```

