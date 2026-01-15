import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
import matplotlib.pyplot as plt


from src.classical.cnn import PCamCNN
from src.data.pcam_loader import get_pcam_dataset
from src.data.transforms import get_train_transforms, get_eval_transforms

torch.backends.cudnn.benchmark = True

# ----------------------------
# Configuration (keep simple)
# ----------------------------
DATA_ROOT = '/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/dataset'   # CHANGE if needed
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
EMBEDDING_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "results/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------
# Training / Evaluation loops
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ----------------------------
# Main
# ----------------------------
def main():
    print(f"🚀 Training on device: {DEVICE}")

    # Datasets
    train_set = get_pcam_dataset(
        data_dir=DATA_ROOT,
        split="train",
        transform=get_train_transforms()
    )

    val_set = get_pcam_dataset(
        data_dir=DATA_ROOT,
        split="val",
        transform=get_eval_transforms()
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # Model
    model = PCamCNN(embedding_dim=EMBEDDING_DIM).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📘 Epoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, "pcam_cnn_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")

    print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc:.4f}")

    # ----------------------------
    # Save training history
    # ----------------------------
    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ----------------------------
    # Plot curves
    # ----------------------------
    os.makedirs("results/figures", exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("CNN Training Loss")
    plt.savefig("results/figures/cnn_loss.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("CNN Training Accuracy")
    plt.savefig("results/figures/cnn_accuracy.png")
    plt.close()

if __name__ == "__main__":
    main()
