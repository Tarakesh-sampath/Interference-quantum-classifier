import os
import numpy as np

from src.utils.paths import load_paths
from src.utils.seed import set_seed

from src.IQL.encoding.embedding_to_state import embedding_to_state
from src.IQL.models.winner_take_all import WinnerTakeAll
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier
from src.IQL.backends.transition import TransitionBackend
from src.IQL.learning.memory_bank import MemoryBank
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
os.makedirs(EMBED_DIR, exist_ok=True)


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
# 🔒 LOAD MEMORY BANK FROM REGIME 3-C
# -------------------------------------------------
# IMPORTANT:
# This must be the SAME memory_bank produced by Regime 3-C

MEMORY_PATH = os.path.join(PATHS["artifacts"], "regime3c_memory.pkl")

with open(MEMORY_PATH, "rb") as f:
    memory_bank = pickle.load(f)

print("Loaded memory bank with",
      len(memory_bank.class_states),
      "memories")


# -------------------------------------------------
# 🔁 CONSOLIDATION PHASE (NO GROWTH)
# -------------------------------------------------
# Use Regime 3-A trainer:
# - updates memories
# - NO spawning logic
trainer = WinnerTakeAll(
    memory_bank=memory_bank,
    eta=0.05,      # slightly smaller eta for stabilization
    backend=TransitionBackend()
)

acc_train = trainer.fit(X_train, y_train)
print("Consolidation pass accuracy:", acc_train)
print("Updates during consolidation:", trainer.num_updates)


# -------------------------------------------------
# 📊 FINAL EVALUATION (Regime 3-B inference)
# -------------------------------------------------
classifier = WeightedVoteClassifier(memory_bank)

correct = 0
for x, y in zip(X_train, y_train):
    if classifier.predict(x) == y:
        correct += 1

final_acc = correct / len(X_train)
print("FINAL Regime 3-C accuracy:", final_acc)


### output
"""
🌱 Global seed set to 42
Loaded train embeddings: (3500, 32)
Loaded memory bank with 22 memories
Consolidation pass accuracy: 0.8048571428571428
Updates during consolidation: 683
FINAL Regime 3-C accuracy: 0.884
"""
