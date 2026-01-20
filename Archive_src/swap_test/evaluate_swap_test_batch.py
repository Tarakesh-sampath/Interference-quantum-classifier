import os
import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from src.utils.paths import load_paths
from src.utils.seed import set_seed


# ----------------------------
# Reproducibility
# ----------------------------
set_seed(42)

# ----------------------------
# Load paths
# ----------------------------
BASE_ROOT, PATHS = load_paths()
EMBED_DIR = PATHS["embeddings"]

# ----------------------------
# Quantum-safe conversion
# ----------------------------
def to_quantum_state(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if not (n & (n - 1) == 0):
        raise ValueError(f"State length {n} is not power of 2")
    x = x / np.sqrt(np.sum(x ** 2))
    assert np.isclose(np.sum(x ** 2), 1.0, atol=1e-12)
    return x


# ----------------------------
# Load class states
# ----------------------------
class_state_0 = to_quantum_state(
    np.load(os.path.join(EMBED_DIR, "class_state_0.npy"))
)
class_state_1 = to_quantum_state(
    np.load(os.path.join(EMBED_DIR, "class_state_1.npy"))
)

# ----------------------------
# Load test embeddings
# ----------------------------
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

# ----------------------------
# Evaluation subset
# ----------------------------
N_SAMPLES = 5000
SHOTS = 1024

#X = X[:N_SAMPLES]
#y = y[:N_SAMPLES]

# ----------------------------
# SWAP test fidelity
# ----------------------------
def swap_test_fidelity(state_a, state_b, shots=1024):
    n_qubits = int(np.log2(len(state_a)))
    qc = QuantumCircuit(1 + 2 * n_qubits, 1)

    anc = 0
    reg_a = list(range(1, 1 + n_qubits))
    reg_b = list(range(1 + n_qubits, 1 + 2 * n_qubits))

    qc.initialize(state_a, reg_a)
    qc.initialize(state_b, reg_b)

    qc.h(anc)
    for qa, qb in zip(reg_a, reg_b):
        qc.cswap(anc, qa, qb)
    qc.h(anc)

    qc.measure(anc, 0)

    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    p0 = counts.get("0", 0) / shots
    fidelity = 2 * p0 - 1
    return fidelity


# ----------------------------
# Batch evaluation
# ----------------------------
correct = 0

print(f"\n🔬 Evaluating SWAP-test classifier on {N_SAMPLES} samples\n")

for i in tqdm(range(N_SAMPLES)):
    x = to_quantum_state(X[i])

    F0 = swap_test_fidelity(x, class_state_0, shots=SHOTS)
    F1 = swap_test_fidelity(x, class_state_1, shots=SHOTS)

    pred = 0 if F0 > F1 else 1
    if pred == y[i]:
        correct += 1

accuracy = correct / N_SAMPLES

print("\n==============================")
print("Measurement-based Quantum SWAP Test")
print(f"Samples: {N_SAMPLES}")
print(f"Shots per test: {SHOTS}")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")

## output
"""
🌱 Global seed set to 42

🔬 Evaluating SWAP-test classifier on 5000 samples

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [03:46<00:00, 22.11it/s]

==============================
Measurement-based Quantum SWAP Test
Samples: 5000
Shots per test: 1024
Accuracy: 0.8784
==============================
"""
