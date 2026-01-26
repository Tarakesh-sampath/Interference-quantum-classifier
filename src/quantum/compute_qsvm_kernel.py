import os
import json
import numpy as np
from tqdm import tqdm

from qiskit_aer.primitives import SamplerV2
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute

from src.utils.paths import load_paths
from src.utils.seed import set_seed

# ------------------------------------------------------------
# Reproducibility
# --------------------------------------------
set_seed(42)

# ------------------------------------------------------------
# Load paths and data
# ------------------------------------------------------------
BASE_ROOT, PATHS = load_paths()

EMBED_DIR = PATHS["embeddings"]
OUT_DIR = os.path.join(BASE_ROOT, "results", "qsvm_cache")
os.makedirs(OUT_DIR, exist_ok=True)

X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
test_idx  = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# ------------------------------------------------------------
# SUBSAMPLING for Baseline Efficiency
# ------------------------------------------------------------
# Limiting to 500 samples because O(N^2) kernel computation 
# for 3500 samples would take ~17 hours on GPU.
MAX_TRAIN = 500000
MAX_TEST  = 200000

if len(X_train) > MAX_TRAIN:
    print(f"Subsampling train set from {len(X_train)} to {MAX_TRAIN}...")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_train), MAX_TRAIN, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

if len(X_test) > MAX_TEST:
    print(f"Subsampling test set from {len(X_test)} to {MAX_TEST}...")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_test), MAX_TEST, replace=False)
    X_test = X_test[indices]
    y_test = y_test[indices]

# ------------------------------------------------------------
# Normalize embeddings
# ------------------------------------------------------------
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test  = X_test  / np.linalg.norm(X_test, axis=1, keepdims=True)

# Infer number of qubits
dim = X_train.shape[1]
num_qubits = int(np.log2(dim))
assert 2 ** num_qubits == dim, "Embedding dimension must be 2^n"

# ------------------------------------------------------------
# Define FIXED quantum feature map
# ------------------------------------------------------------
feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=1,
    entanglement="linear"
)

# ------------------------------------------------------------
# GPU Accelerated Backend (Aer SamplerV2)
# ------------------------------------------------------------
sampler = SamplerV2(
    options={"backend_options": {"method": "statevector", "device": "GPU"}}
)
fidelity = ComputeUncompute(sampler=sampler)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    fidelity=fidelity
)

# ------------------------------------------------------------
# Compute and save TRAIN kernel
# ------------------------------------------------------------
print(f"Computing QSVM TRAIN kernel ({len(X_train)}x{len(X_train)})...")
K_train = quantum_kernel.evaluate(X_train, X_train)
np.save(os.path.join(OUT_DIR, "qsvm_kernel_train.npy"), K_train)

# ------------------------------------------------------------
# Compute and save TEST kernel
# ------------------------------------------------------------
print(f"Computing QSVM TEST kernel ({len(X_test)}x{len(X_train)})...")
K_test = quantum_kernel.evaluate(X_test, X_train)
np.save(os.path.join(OUT_DIR, "qsvm_kernel_test.npy"), K_test)

# ------------------------------------------------------------
# Save Labels for verification
# ------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "y_train_sub.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test_sub.npy"), y_test)

# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------
meta = {
    "model": "QSVM",
    "num_qubits": num_qubits,
    "num_train": int(X_train.shape[0]),
    "num_test": int(X_test.shape[0]),
    "embedding_dimension": int(dim),
    "subsampling": True
}

with open(os.path.join(OUT_DIR, "qsvm_kernel_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("QSVM kernel computation complete.")
