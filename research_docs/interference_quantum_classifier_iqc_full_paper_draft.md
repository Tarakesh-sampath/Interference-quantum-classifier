# Interference Quantum Classifier (IQC)

## A Measurement‑Efficient Quantum Classification Framework Based on Linear Interference

---

## Abstract

Quantum machine learning classifiers proposed for near‑term quantum hardware are commonly formulated as variational models or similarity‑based methods relying on probability or fidelity estimation. While theoretically expressive, such approaches often incur high measurement cost, unstable optimization dynamics, and loss of phase information, limiting their practical applicability on noisy intermediate‑scale quantum (NISQ) devices. In this work, we present the **Interference Quantum Classifier (IQC)**, a hybrid quantum–classical classification framework in which learning is decoupled from quantum execution and inference is performed through a fixed quantum interference procedure. IQC derives its decision signal from a linear interference quantity rather than a quadratic probability measure, enabling sign‑sensitive and phase‑aware classification with constant measurement complexity. We develop the formal mathematical foundations of the framework, introduce an interference‑based decision observable, and describe learning as state evolution in Hilbert space carried out entirely in classical computation. Experimental evaluations on medical image embeddings demonstrate stable behavior across learning regimes, robustness to measurement noise, and favorable runtime characteristics relative to measurement‑heavy quantum baselines. These results suggest that linear quantum interference provides a viable and interpretable primitive for quantum classification in near‑term settings.

---

## 1. Introduction

Quantum machine learning (QML) has been widely explored as a potential application domain for near‑term quantum computers. Proposed quantum classifiers range from variational quantum circuits trained by measurement‑based optimization to kernel and similarity methods that estimate quantum state overlap. Despite promising theoretical constructions, many such approaches face significant practical challenges, including large sampling overhead, sensitivity to noise, and limited interpretability.

A common feature of existing quantum classifiers is their reliance on **quadratic observables**, such as probabilities or fidelities, as the basis for decision making. While natural from a measurement perspective, these quantities discard sign and relative phase information and typically require repeated circuit executions to estimate reliably. Moreover, when combined with variational training, they introduce optimization pathologies such as barren plateaus.

In this work, we explore an alternative design philosophy for quantum classification. Rather than treating the quantum circuit as a trainable model, we treat it as a **fixed physical instrument** that evaluates an interference‑based quantity between quantum state representations. Learning is performed outside the quantum circuit by updating class‑representative states, while inference is realized through a constant‑depth interference procedure. This perspective motivates the Interference Quantum Classifier (IQC).

The contributions of this paper are threefold. First, we formalize a linear interference quantity as a decision primitive for classification and analyze its geometric and physical properties. Second, we describe a learning framework in which class information is accumulated as quantum state evolution without in‑circuit optimization. Third, we empirically evaluate the resulting classifier across multiple learning regimes, demonstrating stable and measurement‑efficient behavior consistent with the theoretical design.

---

## 2. Problem Setup and Notation

We consider supervised binary classification tasks. Input data are mapped to real‑valued feature vectors using a classical representation model, such as a convolutional neural network. These feature vectors are normalized and deterministically encoded into quantum states.

Let \(\mathcal{H} = \mathbb{C}^{2^n}\) denote a finite‑dimensional Hilbert space. An input sample is represented by a normalized quantum state \(|\psi\rangle \in \mathcal{H}\). Class information is represented by one or more normalized quantum states \(|\chi\rangle \in \mathcal{H}\), referred to as class states. The goal of classification is to assign a label based on the relationship between \(|\psi\rangle\) and \(|\chi\rangle\).

---

## 3. Mathematical Foundations of Linear Interference

### 3.1 Linear and Quadratic State Similarity

Given two quantum states \(|\psi\rangle\) and \(|\chi\rangle\), their inner product \(\langle \chi | \psi \rangle\) defines a complex‑valued linear overlap. In contrast, commonly used similarity measures such as fidelity depend on the squared magnitude \(|\langle \chi | \psi \rangle|^2\), which is quadratic in the state amplitudes.

The linear overlap preserves sign and relative phase information, whereas quadratic measures do not. As a result, the two quantities induce fundamentally different decision geometries in Hilbert space. IQC is built around the observation that classification decisions can be based on linear interference rather than quadratic similarity.

### 3.2 Decision Geometry

Fixing a reference state \(|\chi\rangle\), the real part of the overlap
\[
 f_{\chi}(|\psi\rangle) = \mathrm{Re}\langle \chi | \psi \rangle
\]
defines a linear functional on \(\mathcal{H}\). The decision boundary \(f_{\chi}(|\psi\rangle)=0\) corresponds to a hyperplane in Hilbert space, analogous to linear classifiers in classical learning theory.

---

## 4. Interference‑Based Decision Observable

The quantity \(\mathrm{Re}\langle \chi | \psi \rangle\) cannot be obtained from a single‑state measurement, as expectation values of Hermitian operators are quadratic in the state amplitudes. To access this linear quantity physically, IQC employs **quantum interference**.

An ancilla‑assisted interference procedure prepares a coherent superposition in which branches associated with \(|\psi\rangle\) and \(|\chi\rangle\) interfere. Measurement of the ancilla converts relative phase and overlap into a scalar signal whose expectation value equals the desired linear quantity. The sign of this signal serves as the classification decision.

Importantly, the interference procedure is fixed and does not depend on learned parameters or dataset size. It therefore constitutes a measurement‑efficient and hardware‑agnostic inference mechanism.

---

## 5. Learning as Quantum State Evolution

IQC performs learning by updating the classical description of the class state \(|\chi\rangle\). Given a labeled training sample \((|\psi\rangle, y)\) with \(y \in \{+1,-1\}\), the class state is updated according to
\[
 |\chi'\rangle = \frac{|\chi\rangle + \eta y |\psi\rangle}{\| |\chi\rangle + \eta y |\psi\rangle \|},
\]
where \(\eta\) is a learning rate.

This update corresponds to a projection onto the unit sphere in Hilbert space and adjusts the orientation of the decision hyperplane to increase the signed interference score for correctly labeled samples. No quantum gradients or parameterized circuits are involved. Stochastic variants of this update accommodate noise and finite‑shot effects without altering the inference mechanism.

---

## 6. Learning Regimes

The IQC framework admits multiple learning paradigms built upon the same interference‑based inference:

1. **Static regime:** a class state is constructed offline by aggregating labeled samples.
2. **Online regime:** the class state evolves incrementally as new data arrive.
3. **Multi‑state regime:** multiple class states are maintained and combined through classical aggregation.

Across all regimes, the quantum circuit and decision observable remain invariant. Differences in behavior arise solely from how class information is represented and updated.

---

## 7. Experimental Evaluation

### 7.1 Setup

We evaluated IQC on binary classification tasks derived from medical image datasets. Images were embedded using a fixed convolutional neural network, and the resulting feature vectors were encoded into quantum states. All quantum inference was simulated under consistent noise and shot conditions.

Baselines included a classical linear classifier operating on the same embeddings, a variational quantum classifier, and a fidelity‑based quantum similarity classifier.

### 7.2 Results

Across learning regimes, IQC exhibited stable classification behavior with low variance across repeated inference runs. Increasing shot count alone did not significantly improve the performance of measurement‑based baselines, whereas IQC performance remained robust across a wide range of measurement settings.

The multi‑state regime improved robustness to outliers and heterogeneous data distributions without increasing quantum circuit depth. Variational baselines showed sensitivity to initialization and hyperparameter choices not observed in IQC.

### 7.3 Interpretation

These observations are consistent with the theoretical framework: IQC’s reliance on interference yields a low‑variance decision signal, and learning outside the quantum circuit avoids optimization‑induced instability. Performance limitations were primarily attributable to the quality of classical embeddings rather than quantum execution.

---

## 8. Discussion

IQC highlights a different role for quantum circuits in machine learning. Rather than serving as trainable models, quantum circuits act as fixed physical operators that evaluate structured similarity through interference. This perspective leads to reduced measurement cost, improved stability, and clearer interpretability.

At the same time, IQC inherits limitations of linear classifiers: when class separation is not achievable in the chosen representation space, performance degrades. Addressing this limitation requires improvements in feature extraction or representational diversity rather than deeper quantum circuits.

---

## 9. Conclusion

We have presented the Interference Quantum Classifier, a quantum classification framework based on linear interference and a strict separation between learning and inference. By avoiding variational training and quadratic similarity estimation, IQC provides a measurement‑efficient and interpretable approach to quantum classification compatible with near‑term hardware. Our theoretical and empirical results suggest that quantum interference, when used as a decision primitive, offers a promising and underexplored pathway for practical quantum machine learning.

---

## Acknowledgements

The authors acknowledge helpful discussions and publicly available datasets that made this study possible.

