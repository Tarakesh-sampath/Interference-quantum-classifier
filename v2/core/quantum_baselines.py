"""Measurement-based quantum baselines (QSVM, VQC) as v2 model wrappers.

Reuse the amplitude-encoding setup from src/quantum/* (RawFeatureVector +
FidelityStatevectorKernel for QSVM; RawFeatureVector + 1-layer RY/CX ansatz +
COBYLA for VQC). These are statevector-exact at fit time; shot behavior for the
sweep is handled separately. Inputs are assumed already L2-normalized.
"""

from __future__ import annotations

import numpy as np

from v2.core.seeding import set_global_seed


class QSVMWrapper:
    is_iqc = False

    def __init__(self, params, seed):
        self.params = params or {}
        self.seed = seed
        self.clf = None

    def fit(self, X, y_bin):
        set_global_seed(self.seed)
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit_machine_learning.circuit.library import RawFeatureVector
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel

        dim = X.shape[1]
        fm = RawFeatureVector(feature_dimension=dim)
        qkernel = FidelityStatevectorKernel(feature_map=fm)
        self.clf = QSVC(quantum_kernel=qkernel)
        self.clf.fit(X, y_bin)
        return self

    def decision_scores(self, X):
        try:
            return self.clf.decision_function(X)
        except Exception:
            return self.clf.predict(X).astype(float)

    def predict(self, X):
        return self.clf.predict(X).astype(int)


class VQCWrapper:
    is_iqc = False

    def __init__(self, params, seed):
        self.params = params or {}
        self.seed = seed
        self.clf = None

    def fit(self, X, y_bin):
        set_global_seed(self.seed)
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.primitives import Sampler
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.circuit.library import RawFeatureVector
        from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss

        try:
            from qiskit_algorithms.optimizers import COBYLA
        except ImportError:
            from qiskit.algorithms.optimizers import COBYLA

        dim = X.shape[1]
        nq = int(np.log2(dim))
        fm = RawFeatureVector(feature_dimension=dim)
        theta = ParameterVector("θ", length=nq)
        ansatz = QuantumCircuit(nq)
        for i in range(nq):
            ansatz.ry(theta[i], i)
        for i in range(nq - 1):
            ansatz.cx(i, i + 1)
        ansatz.cx(nq - 1, 0)

        rng = np.random.default_rng(self.seed)
        init = rng.uniform(-np.pi, np.pi, size=nq)
        self.clf = VQC(sampler=Sampler(), feature_map=fm, ansatz=ansatz,
                       optimizer=COBYLA(maxiter=self.params.get("maxiter", 100)),
                       loss=CrossEntropyLoss(), initial_point=init)
        self.clf.fit(X, y_bin)
        return self

    def decision_scores(self, X):
        # VQC exposes class probabilities via the neural network forward pass
        try:
            probs = self.clf.neural_network.forward(X, self.clf.weights)
            return probs[:, 1] - probs[:, 0]
        except Exception:
            return self.clf.predict(X).astype(float)

    def predict(self, X):
        preds = self.clf.predict(X)
        preds = np.asarray(preds)
        if preds.ndim > 1:  # one-hot
            preds = preds.argmax(1)
        return preds.astype(int)
