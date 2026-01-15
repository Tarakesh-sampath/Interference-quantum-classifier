import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.classical.cnn import PCamCNN
from src.data.pcam_loader import get_pcam_dataset
from src.data.transforms import get_eval_transforms

# ----------------------------
# Config
# ----------------------------
DATA_ROOT = '/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/dataset'   # adjust if needed
CHECKPOINT = "../results/checkpoints/pcam_cnn_best.pt"
EMBEDDING_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "../results/embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_SAMPLES = 5000   # enough for visualization, fast on GTX 1650
BATCH_SIZE = 128

print(f"🚀 Using device: {DEVICE}")

# ----------------------------
# Load model
# ----------------------------
model = PCamCNN(embedding_dim=EMBEDDING_DIM).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ----------------------------
# Dataset (validation split is best for inspection)
# ----------------------------
dataset = get_pcam_dataset(
    data_dir=DATA_ROOT,
    split="val",
    transform=get_eval_transforms()
)

subset = Subset(dataset, range(MAX_SAMPLES))
loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=6,
    pin_memory=True
)

# ----------------------------
# Extract embeddings
# ----------------------------
all_embeddings = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(DEVICE)
        embeddings = model(images, return_embedding=True)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

embeddings = np.concatenate(all_embeddings, axis=0)
labels = np.concatenate(all_labels, axis=0)

# ----------------------------
# Save
# ----------------------------
np.save(os.path.join(SAVE_DIR, "val_embeddings.npy"), embeddings)
np.save(os.path.join(SAVE_DIR, "val_labels.npy"), labels)

print("✅ Embeddings saved:")
print(" - val_embeddings.npy", embeddings.shape)
print(" - val_labels.npy", labels.shape)
