import os
import numpy as np
from tqdm import tqdm

from qiskit.quantum_info import Statevector

from src.utils.paths import load_paths
from src.utils.seed import set_seed


# ----------------------------
# Reproducibility
# ----------------------------
set_seed(42)

# ----------------------------
# Load paths
# ----------------------------
_, PATHS = load_paths()
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
phi0 = to_quantum_state(
    np.load(os.path.join(EMBED_DIR, "class_state_0.npy"))
)
phi1 = to_quantum_state(
    np.load(os.path.join(EMBED_DIR, "class_state_1.npy"))
)

sv_phi0 = Statevector(phi0)
sv_phi1 = Statevector(phi1)


# ----------------------------
# Load embeddings
# ----------------------------
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))
test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

X = X[test_idx]
y = y[test_idx]

N = len(X)
correct = 0

print(f"\n🔬 Evaluating measurement-free statevector classifier on {N} samples\n")

for i in tqdm(range(N)):
    psi = Statevector(to_quantum_state(X[i]))

    F0 = abs(psi.inner(sv_phi0)) ** 2
    F1 = abs(psi.inner(sv_phi1)) ** 2

    pred = 0 if F0 > F1 else 1
    if pred == y[i]:
        correct += 1

accuracy = correct / N

print("\n==============================")
print("Measurement-free (Statevector) Quantum Classifier")
print(f"Samples: {N}")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")

## output
"""
🌱 Global seed set to 42

🔬 Evaluating measurement-free statevector classifier on 1500 samples

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:00<00:00, 29429.03it/s]

==============================
Measurement-free (Statevector) Quantum Classifier
Samples: 1500
Accuracy: 0.8827
==============================
"""