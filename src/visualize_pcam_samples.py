import matplotlib.pyplot as plt
from src.data.pcam_loader import get_pcam_dataset
from src.utils.paths import load_paths
from src.utils.seed import set_seed

set_seed(42)

_, PATHS = load_paths()
dataset = get_pcam_dataset(PATHS["dataset"], "test")

benign_images = []
malignant_images = []

for i in range(len(dataset)):
    img, label = dataset[i]
    if label == 0 and len(benign_images) < 5:
        benign_images.append(img)
    elif label == 1 and len(malignant_images) < 5:
        malignant_images.append(img)
    if len(benign_images) == 5 and len(malignant_images) == 5:
        break

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, img in enumerate(benign_images):
    axes[0, i].imshow(img.permute(1, 2, 0))
    axes[0, i].set_title("Benign", fontsize=12)
    axes[0, i].axis("off")

for i, img in enumerate(malignant_images):
    axes[1, i].imshow(img.permute(1, 2, 0))
    axes[1, i].set_title("Malignant", fontsize=12)
    axes[1, i].axis("off")

plt.suptitle("PCAM Dataset: Benign (top) vs Malignant (bottom)", fontsize=14)
plt.tight_layout()
plt.savefig("pcam_samples.png", dpi=150)
print("Saved to pcam_samples.png")
plt.show()
