import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.utils.paths import load_paths
from src.utils.seed import set_seed

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
set_seed(42)

# -------------------------------
# Load embeddings
# -------------------------------
_, PATHS = load_paths()
EMBED_DIR = PATHS["embeddings"]
PROTO_BASE = PATHS["class_prototypes"]

X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
y = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))
test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

X_test = X[test_idx]
y_test = y[test_idx]

# -------------------------------
# Select states
# -------------------------------
chi = np.load(os.path.join(PROTO_BASE, "K1/class0_proto0.npy"))
chi = chi / np.linalg.norm(chi)

psi_pos = X_test[y_test == 1][0]
psi_neg = X_test[y_test == 0][0]

psi_pos = psi_pos / np.linalg.norm(psi_pos)
psi_neg = psi_neg / np.linalg.norm(psi_neg)

# Stack for projection
states = np.vstack([chi, psi_pos, psi_neg])

# -------------------------------
# PCA projection (for visualization only)
# -------------------------------
pca = PCA(n_components=2)
states_2d = pca.fit_transform(states)

chi_2d, psi1_2d, psi2_2d = states_2d

# =====================================================
# FIGURE 3(a): Interference Geometry
# =====================================================
plt.figure(figsize=(5, 5))

# Decision hyperplane normal
plt.axline((0, 0), chi_2d, linestyle="--", linewidth=1)

plt.scatter(*chi_2d, s=80, label="|χ⟩")
plt.scatter(*psi1_2d, s=80, label="|ψ₁⟩ (+)")
plt.scatter(*psi2_2d, s=80, label="|ψ₂⟩ (−)")

plt.text(*chi_2d, "  |χ⟩")
plt.text(*psi1_2d, "  |ψ₁⟩")
plt.text(*psi2_2d, "  |ψ₂⟩")

plt.title("Interference-Based Decision Geometry")
plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()

# =====================================================
# FIGURE 3(b): Fidelity Geometry
# =====================================================
plt.figure(figsize=(5, 5))

# Equal-fidelity contour
r = np.linalg.norm(psi1_2d - chi_2d)
theta = np.linspace(0, 2 * np.pi, 300)

circle_x = chi_2d[0] + r * np.cos(theta)
circle_y = chi_2d[1] + r * np.sin(theta)

plt.plot(circle_x, circle_y, linestyle=":", label="Equal fidelity")

plt.scatter(*chi_2d, s=80, label="|χ⟩")
plt.scatter(*psi1_2d, s=80, label="|ψ₁⟩")
plt.scatter(*psi2_2d, s=80, label="|ψ₂⟩")

plt.text(*chi_2d, "  |χ⟩")
plt.text(*psi1_2d, "  |ψ₁⟩")
plt.text(*psi2_2d, "  |ψ₂⟩")

plt.title("Fidelity-Based Similarity Geometry")
plt.xlabel("PCA dimension 1")
plt.ylabel("PCA dimension 2")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
