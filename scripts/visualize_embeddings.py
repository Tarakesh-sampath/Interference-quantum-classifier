import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os   

BASE_ROOT = '/home/tarakesh/Work/Repo/measurement-free-quantum-classifier'

# ----------------------------
# Load embeddings
# ----------------------------
embeddings = np.load(os.path.join(BASE_ROOT, "results/embeddings/val_embeddings.npy"))
labels = np.load(os.path.join(BASE_ROOT, "results/embeddings/val_labels.npy"))

print("Loaded embeddings:", embeddings.shape)

# ----------------------------
# t-SNE
# ----------------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    init="pca"
)

emb_2d = tsne.fit_transform(embeddings)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7, 6))

plt.scatter(
    emb_2d[labels == 0, 0],
    emb_2d[labels == 0, 1],
    s=8,
    alpha=0.6,
    label="Benign"
)

plt.scatter(
    emb_2d[labels == 1, 0],
    emb_2d[labels == 1, 1],
    s=8,
    alpha=0.6,
    label="Malignant"
)

plt.legend()
plt.title("t-SNE of CNN Embeddings (Validation Set)")
plt.tight_layout()

os.makedirs(os.path.join(BASE_ROOT, "results/figures"), exist_ok=True)
plt.savefig(os.path.join(BASE_ROOT, "results/figures/embedding_tsne.png"), dpi=300)
plt.show()
