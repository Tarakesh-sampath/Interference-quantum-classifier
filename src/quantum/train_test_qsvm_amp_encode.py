import os
import json
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

from qiskit_machine_learning.algorithms import QSVC

from src.utils.paths import load_paths
from src.utils.seed import set_seed

# ============================================================
# 0. Reproducibility
# ============================================================
set_seed()


# ============================================================
# 1. Load paths and embeddings
# ============================================================
BASE_ROOT, PATHS = load_paths()
EMBED_DIR = PATHS["embeddings"]
OUT_DIR = os.path.join(BASE_ROOT, "results", "qsvm")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "qsvm_amp_model.dill")

print("Loading embeddings...")
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
test_idx  = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

# Subsample for feasibility
TRAIN_SIZE = 100
TEST_SIZE  = 50

X_train = X[train_idx]#[:TRAIN_SIZE]
y_train = y[train_idx]#[:TRAIN_SIZE]
X_test  = X[test_idx]#[:TEST_SIZE]
y_test  = y[test_idx]#[:TEST_SIZE]

print(f"Original Shape: {X_train.shape}")

# ============================================================
# 2. Preprocessing for Amplitude Encoding
# ============================================================
# Amplitude encoding requires the input vector to be normalized (L2 norm = 1)
print("Normalizing features (L2) for Amplitude Encoding...")
# Using sklearn Normalizer to ensure L2 norm is exactly 1
normalizer = Normalizer(norm='l2')
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm  = normalizer.transform(X_test)

dim = X_train.shape[1]
num_qubits = int(np.log2(dim))
assert 2**num_qubits == dim, f"Dimension {dim} must be a power of 2 for amplitude encoding (2^n)"

print(f"Using {num_qubits} qubits to encode {dim} features.")

# ============================================================
# 3. Define Amplitude Encoding Feature Map using RawFeatureVector
# ============================================================
from qiskit_machine_learning.circuit.library import RawFeatureVector

# RawFeatureVector implements amplitude encoding and handles parameter binding correctly
feature_map = RawFeatureVector(feature_dimension=dim)

# ============================================================
# 4. Quantum Kernel Setup
# ============================================================
print("Setting up FidelityStatevectorKernel for Amplitude Encoding...")
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

# FidelityStatevectorKernel calculates |<psi(x)|psi(y)>|^2 directly using statevectors.
# It does NOT require circuit inversion, so it works with RawFeatureVector/Amplitude Encoding.
qkernel = FidelityStatevectorKernel(feature_map=feature_map)

if os.path.exists(MODEL_PATH) and 1:
    print(f"Loading saved model from {MODEL_PATH} (skipping training)...")
    qsvm = joblib.load(MODEL_PATH)
    train_time = 0.0
else:
    print("Training QSVC (Amplitude Encoding)...")
    start_time = time.time()

    qsvm = QSVC(quantum_kernel=qkernel)
    qsvm.fit(X_train_norm, y_train)

    end_time = time.time()
    train_time = end_time - start_time
    print(f"Training time: {train_time:.4f}s")
    print(f"Time per sample: {train_time / len(X_train_norm):.4f}s")

    qsvm.save(MODEL_PATH)  
    print(f"Model saved to {MODEL_PATH}")

# ============================================================
# 5. Evaluate and Save
# ============================================================
print("Predicting on test set...")
start_time = time.time()
y_pred = qsvm.predict(X_test_norm)
end_time = time.time()
test_time = end_time - start_time
print(f"Test time: {test_time:.4f}s")
print(f"Time per sample: {test_time / len(X_test_norm):.4f}s")

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print(f"QSVC (Amplitude Encoding) Test Accuracy: {accuracy:.4f}")
print("=" * 60)

# Save Results
results = {
    "accuracy": float(accuracy),
    "num_train": len(X_train),
    "num_test": len(X_test),
    "num_features": dim,
    "num_qubits": num_qubits,
    "encoding": "Amplitude Encoding",
    "training_time": train_time
}

out_path = os.path.join(OUT_DIR, "qsvm_amp_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {out_path}")

"""
Loading embeddings...
Original Shape: (3500, 32)
Normalizing features (L2) for Amplitude Encoding...
Using 5 qubits to encode 32 features.
Setting up FidelityStatevectorKernel for Amplitude Encoding...
Training QSVC (Amplitude Encoding)...
Training time: 79.7180s
Time per sample: 0.0228s
Predicting on test set...
Test time: 37.7612s
Time per sample: 0.0252s
============================================================
QSVC (Amplitude Encoding) Test Accuracy: 0.9093
============================================================
Saved results to /home/tarakesh/Work/Repo/measurement-free-quantum-classifier/results/qsvm_final/qsvm_amp_results.json
"""