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
CLASS_DIR = PATHS["class_prototypes"]
EMBED_DIR = PATHS["embeddings"] 

K = int(PATHS["class_count"]["K"])
INDEX_DIM = K
DATA_DIM = 32


# ----------------------------
# Helper
# ----------------------------
def to_quantum_state(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x / np.sqrt(np.sum(x ** 2))
    return x


# ----------------------------
# Load prototypes
# ----------------------------
def load_class_superposition(cls):
    protos = []
    for k in range(1,K):
        p = np.load(os.path.join(CLASS_DIR, f"K{cls}/class{cls}_proto{k}.npy"))
        protos.append(p)

    # Build joint state |k> |phi_k>
    joint = np.zeros(INDEX_DIM * DATA_DIM, dtype=np.float64)

    for k, proto in enumerate(protos):
        joint[k * DATA_DIM:(k + 1) * DATA_DIM] = proto

    joint = joint / np.sqrt(K)  # superposition normalization
    joint = to_quantum_state(joint)

    return Statevector(joint)


# ----------------------------
# Load class states
# ----------------------------
Phi0 = load_class_superposition(0)
Phi1 = load_class_superposition(1)


# ----------------------------
# Load data
# ----------------------------
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))

test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

X = X[test_idx]
y = y[test_idx]
    
N = len(X)
correct = 0

print(f"\n🔬 Evaluating Phase B (K={K}) on {N} samples\n")

# ----------------------------
# Evaluation
# ----------------------------
for i in tqdm(range(N)):
    psi = to_quantum_state(X[i])

    # Lift test state into joint space
    joint_test = np.zeros(INDEX_DIM * DATA_DIM, dtype=np.float64)
    for k in range(K):
        joint_test[k * DATA_DIM:(k + 1) * DATA_DIM] = psi

    joint_test = to_quantum_state(joint_test)
    Psi = Statevector(joint_test)

    F0 = abs(Psi.inner(Phi0)) ** 2
    F1 = abs(Psi.inner(Phi1)) ** 2

    pred = 0 if F0 > F1 else 1
    if pred == y[i]:
        correct += 1

accuracy = correct / N

print("\n==============================")
print("Phase B: Interference-Based Measurement-Free Classifier")
print(f"Prototypes per class: {K}")
print(f"Samples: {N}")
print(f"Accuracy: {accuracy:.4f}")
print("==============================\n")


## output 
"""
🌱 Global seed set to 42

🔬 Evaluating Phase B (K=5) on 1500 samples

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:00<00:00, 35004.26it/s]

==============================
Phase B: Interference-Based Measurement-Free Classifier
Prototypes per class: 5
Samples: 1500
Accuracy: 0.8840
==============================
"""