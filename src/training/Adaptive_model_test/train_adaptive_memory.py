import os
import numpy as np
from collections import Counter

from src.utils.paths import load_paths
from src.utils.seed import set_seed

from src.IQL.states.class_state import ClassState
from src.IQL.encoding.embedding_to_state import embedding_to_state
from src.IQL.memory.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend

from src.IQL.models.adaptive_memory import AdaptiveMemory
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier
import pickle


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
set_seed(42)


# -------------------------------------------------
# Load paths
# -------------------------------------------------
_, PATHS = load_paths()
EMBED_DIR = PATHS["embeddings"]
MEMORY_PATH = os.path.join(PATHS["artifacts"], "regime3c_memory.pkl")

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(PATHS["artifacts"], exist_ok=True)

# -------------------------------------------------
# Load embeddings (TRAIN SPLIT)
# -------------------------------------------------
X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels_polar.npy"))
train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))

X_train = X[train_idx]
y_train = y[train_idx]

print("Loaded train embeddings:", X_train.shape)


# -------------------------------------------------
# Initialize memory bank (M = 3)
# -------------------------------------------------
d = X_train[0].shape[0]

class_states = []
for _ in range(3):
    v = np.random.randn(d)
    v /= np.linalg.norm(v)
    class_states.append(ClassState(v))

backend = ExactBackend()

memory_bank = MemoryBank(
    class_states=class_states,
    backend=backend
)

print("Initial number of memories:", len(memory_bank.class_states))


# -------------------------------------------------
# Train Regime 3-C (percentile-based τ)
# -------------------------------------------------
trainer = AdaptiveMemory(
    memory_bank=memory_bank,
    eta=0.1,
    percentile=5,       # τ = 5th percentile of margins
    tau_abs = -0.121,
    margin_window=500,   # sliding window for stability
    backend=backend,
)

trainer.fit(X_train, y_train)

print("Training finished.")
print("Number of memories after training:", len(memory_bank.class_states))
print("Number of spawned memories:", trainer.num_spawns)
print("Number of updates:", trainer.num_updates)


# -------------------------------------------------
# Evaluate using Regime 3-B inference
# -------------------------------------------------
classifier = WeightedVoteClassifier(memory_bank)

correct = 0
for psi, y in zip(X_train, y_train):
    if classifier.predict(psi) == y:
        correct += 1

acc_3c = correct / len(X_train)
print("Regime 3-C accuracy (3-B inference):", acc_3c)


# -------------------------------------------------
# Optional diagnostics
# -------------------------------------------------
print("Final memory count:", len(memory_bank.class_states))

with open(MEMORY_PATH, "wb") as f:
    pickle.dump(memory_bank, f)

print("Saved Regime 3-C memory bank.")

### output
"""
🌱 Global seed set to 42
Loaded train embeddings: (3500, 32)
Initial number of memories: 3
Training finished.
Number of memories after training: 22
Number of spawned memories: 19
Number of updates: 429
Regime 3-C accuracy (3-B inference): 0.788
Final memory count: 22
Saved Regime 3-C memory bank.
"""