"""Retrain PCamCNN with the *correct* transforms, one checkpoint per seed.

v1 silently dropped the transforms (positional-arg bug in get_pcam_dataset), so
the CNN saw un-normalized [0,1] tensors with no augmentation. That bug is fixed in
src/data/pcam_loader.py; this script trains on PCam `train`, early-stops on `val`,
and writes a per-seed checkpoint under v2/results/cnn/cnn_seed{S}/.

    uv run python -m v2.experiments.train_cnn_v2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classical.cnn import PCamCNN
from src.data.pcam_loader import get_pcam_dataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.utils.paths import load_paths
from v2.core.seeding import set_global_seed
from v2.core.io import REPO_ROOT

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
EMBEDDING_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cnn_dir(seed: int) -> str:
    d = os.path.join(REPO_ROOT, "v2", "results", "cnn", f"cnn_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    from tqdm import tqdm

    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    from tqdm import tqdm

    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main(seed: int):
    set_global_seed(seed)
    _, PATHS = load_paths()
    DATA_ROOT = PATHS["dataset"]
    out = _cnn_dir(seed)
    print(f"CNN retrain seed={seed} device={DEVICE} -> {out}")

    train_set = get_pcam_dataset(DATA_ROOT, "train", get_train_transforms())
    val_set = get_pcam_dataset(DATA_ROOT, "val", get_eval_transforms())

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=6,
                              pin_memory=True, generator=g)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    model = PCamCNN(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc, patience, wait = 0.0, 10, 0
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)
        for k, v in zip(history, [tr_loss, tr_acc, val_loss, val_acc]):
            history[k].append(v)
        print(f"[seed {seed}] epoch {epoch}/{EPOCHS} train_acc {tr_acc:.4f} val_acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out, "pcam_cnn_best.pt"))
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"[seed {seed}] early stopping at epoch {epoch}")
            break

    torch.save(model.state_dict(), os.path.join(out, "pcam_cnn_final.pt"))
    history["best_val_acc"] = best_val_acc
    history["seed"] = seed
    with open(os.path.join(out, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[seed {seed}] done best_val_acc={best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    main(args.seed)
