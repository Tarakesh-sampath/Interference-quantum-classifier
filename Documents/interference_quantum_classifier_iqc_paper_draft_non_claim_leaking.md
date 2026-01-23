# Interference Quantum Classifier (IQC)

## A Measurement-Efficient Quantum Classification Framework Based on Linear Interference

---

## Abstract

Quantum machine learning classifiers proposed for near-term devices commonly rely on variational circuits or fidelity-based measurements, leading to high measurement cost, loss of phase information, and unstable training dynamics. In this work, we introduce the **Interference Quantum Classifier (IQC)**, a classification framework in which learning is decoupled from quantum execution and inference is performed through a fixed quantum interference circuit. IQC bases its decision rule on a linear interference quantity rather than probability or fidelity, enabling phase-sensitive, sign-preserving classification with constant measurement complexity. We present the theoretical formulation of the interference observable, describe a quantum circuit realization, and demonstrate how class information can be represented and updated as quantum states using classical learning rules. Experiments on real-world image embeddings show that interference-based aggregation improves expressivity while significantly reducing runtime compared to measurement-heavy quantum classifiers. Our results suggest that linear quantum interference provides a practical and interpretable alternative to variational and kernel-based quantum classification on near-term hardware.

---

## 1. Introduction

Quantum machine learning (QML) has attracted significant attention as a potential application of near-term quantum devices. Most existing quantum classifiers fall into two categories: variational quantum classifiers, which train parameterized circuits using measurement-based optimization, and similarity-based classifiers, which estimate quantum state fidelity or kernel values. In practice, both approaches face substantial challenges, including high measurement overhead, sensitivity to noise, and limited interpretability.

A key observation motivating this work is that classification decisions need not depend on quadratic probability estimates. Instead, they can be derived from **linear interference between quantum states**, which preserves directional and phase information that is lost in fidelity-based methods. This observation motivates a rethinking of how quantum classifiers are constructed and how learning is integrated with quantum hardware.

In this paper, we propose the Interference Quantum Classifier (IQC), a framework that separates learning from quantum inference and employs a fixed quantum interference circuit as its decision engine.

---

## 2. Problem Setup and Notation

We consider a supervised binary classification problem. Input samples are first mapped to real-valued feature vectors using a classical encoder. These vectors are normalized and embedded into quantum states. Let |ψ⟩ denote a quantum state corresponding to an input sample, and let |χ⟩ denote a quantum state representing class information.

The goal of classification is to determine a label based on the relationship between |ψ⟩ and |χ⟩.

---

## 3. Linear Interference as a Decision Primitive

### 3.1 Interference Observable

IQC is built around a linear interference quantity given by the real part of the inner product between two quantum states. Unlike fidelity, which depends on the squared magnitude of the inner product, this quantity preserves sign and phase information.

We show that this linear quantity is sufficient to define a stable and interpretable decision rule for classification.

---

### 3.2 Comparison with Fidelity-Based Classification

Fidelity-based classifiers estimate |⟨χ|ψ⟩|², which is invariant under global phase changes and discards sign information. As a result, such classifiers behave like distance measures rather than directional similarity measures.

In contrast, linear interference distinguishes between constructive and destructive overlap, enabling sign-sensitive classification decisions.

---

## 4. Quantum Circuit for Interference-Based Inference

We describe a quantum circuit that evaluates the linear interference quantity using an ancilla-assisted interference procedure. The circuit is fixed and does not contain trainable parameters. Its output is a single expectation value whose sign determines the predicted class label.

Importantly, the circuit depth and measurement cost are independent of dataset size.

---

## 5. Learning via Quantum State Representation

Rather than training quantum gate parameters, IQC represents learned class information as quantum states. Learning is performed by updating these state representations using classical rules, while the quantum circuit remains unchanged.

This separation avoids common training pathologies encountered in variational quantum algorithms and enables incremental learning.

---

## 6. Learning Regimes

We outline several learning regimes supported by the IQC framework, including:

- static construction of class states from training data,
- online updates using sequential samples,
- use of multiple class states to increase expressivity.

These regimes differ in how class information is represented but share the same interference-based inference mechanism.

---

## 7. Experimental Evaluation

We evaluate IQC on real-world image embeddings generated by a convolutional neural network. We compare interference-based classification against classical baselines and measurement-heavy quantum similarity methods.

Our experiments demonstrate that removing measurement noise alone does not significantly improve performance, whereas interference-based aggregation improves classification accuracy while reducing runtime by orders of magnitude.

---

## 8. Discussion

The IQC framework highlights a different role for quantum circuits in machine learning: rather than serving as trainable models, they act as fixed physical operators that evaluate structured similarity measures. This perspective offers advantages in stability, interpretability, and hardware compatibility.

We discuss limitations of the current approach and potential extensions, including richer quantum memory structures and alternative interference semantics.

---

## 9. Conclusion

We have presented the Interference Quantum Classifier, a quantum classification framework based on linear quantum interference and a clear separation between learning and inference. By avoiding variational training and fidelity estimation, IQC provides a practical path toward measurement-efficient quantum classification on near-term devices. Our results suggest that linear interference is a powerful and underexplored primitive for quantum machine learning.

---

## Acknowledgements

[To be added]
