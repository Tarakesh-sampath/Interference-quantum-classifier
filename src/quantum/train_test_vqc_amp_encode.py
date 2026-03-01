import os
import json
import numpy as np
import time
import joblib
import dill
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import StatePreparation
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler  # Using reference Sampler for better compatibility with parameterized initialize

from src.utils.paths import load_paths
from src.utils.seed import set_seed


# ============================================================
# 0. Reproducibility
# ============================================================
set_seed()


# ============================================================
# 1. Load Data
# ============================================================
BASE_ROOT, PATHS = load_paths()
EMBED_DIR = PATHS["embeddings"]

OUT_DIR = os.path.join(BASE_ROOT, "results", "vqc_amp_simple")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "vqc_amp_simple.dill")
RESULT_PATH = os.path.join(OUT_DIR, "vqc_amp_simple_results.json")

print("Loading embeddings...")
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
test_idx  = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

X_train = X[train_idx]
y_train = y[train_idx]
X_test  = X[test_idx]
y_test  = y[test_idx]

# ============================================================
# 2. Amplitude Normalization
# ============================================================
normalizer = Normalizer(norm='l2')
X_train = normalizer.fit_transform(X_train)
X_test  = normalizer.transform(X_test)

dim = X_train.shape[1]
num_qubits = int(np.log2(dim))
assert 2**num_qubits == dim, "Feature dimension must be power of 2"

print(f"Using {num_qubits} qubits (Amplitude Encoding)")


# ============================================================
# 3. Define Feature Map
# ============================================================
# Feature map (Amplitude Encoding)
# We use a ParameterVector to represent the features. 
# RawFeatureVector is the standard way, but if it fails, we can try to wrap it.
from qiskit_machine_learning.circuit.library import RawFeatureVector
feature_map = RawFeatureVector(feature_dimension=dim)


# ============================================================
# 4. Define SIMPLE Ansatz
# ============================================================
theta = ParameterVector("θ", length=num_qubits)
ansatz = QuantumCircuit(num_qubits)

# Single rotation layer
for i in range(num_qubits):
    ansatz.ry(theta[i], i)

# Simple entanglement ring
for i in range(num_qubits - 1):
    ansatz.cx(i, i + 1)
ansatz.cx(num_qubits - 1, 0)


# ============================================================
# 5. Optimizer & Sampler
# ============================================================
optimizer = COBYLA(maxiter=100)
sampler = Sampler()


# ============================================================
# 6. Train or Load
# ============================================================
if os.path.exists(MODEL_PATH) and 1:
    print(f"Loading saved model from {MODEL_PATH}...")
    # NOTE: Qiskit's VQC.load often requires the same environment
    vqc = VQC.load(MODEL_PATH)
    vqc.sampler = sampler
    train_time = 0.0
else:
    print("Training simple amplitude-VQC...")

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        loss=CrossEntropyLoss()
    )

    start = time.time()
    vqc.fit(X_train, y_train)
    end = time.time()

    train_time = end - start
    print(f"Training time: {train_time:.2f}s")

    vqc.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# ============================================================
# 7. Evaluate
# ============================================================
print("Evaluating...")
start = time.time()
y_pred = vqc.predict(X_test)
end = time.time()

test_time = end - start
accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print(f"Simple Amplitude-VQC Accuracy: {accuracy:.4f}")
print("=" * 60)


# ============================================================
# 8. Save Results
# ============================================================
results = {
    "accuracy": float(accuracy),
    "num_train": len(X_train),
    "num_test": len(X_test),
    "num_features": dim,
    "num_qubits": num_qubits,
    "encoding": "Amplitude Encoding",
    "ansatz": "Single RY layer + CNOT ring",
    "num_parameters": num_qubits,
    "training_time": train_time,
    "test_time": test_time
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULT_PATH}")


"""
Loading embeddings...
Using 5 qubits (Amplitude Encoding)
/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/src/quantum/train_test_vqc_amp_encode.py:96: DeprecationWarning: The class ``qiskit.primitives.sampler.Sampler`` is deprecated as of qiskit 1.2. It will be removed no earlier than 3 months after the release date. All implementations of the `BaseSamplerV1` interface have been deprecated in favor of their V2 counterparts. The V2 alternative for the `Sampler` class is `StatevectorSampler`.
  sampler = Sampler()
Training simple amplitude-VQC...
Training time: 650.60s
Model saved to /home/tarakesh/Work/Repo/measurement-free-quantum-classifier/results/vqc_amp_simple/vqc_amp_simple.dill
Evaluating...
============================================================
Simple Amplitude-VQC Accuracy: 0.8187
============================================================
Results saved to /home/tarakesh/Work/Repo/measurement-free-quantum-classifier/results/vqc_amp_simple/vqc_amp_simple_results.json
"""