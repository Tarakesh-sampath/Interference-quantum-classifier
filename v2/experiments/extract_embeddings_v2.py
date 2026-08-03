"""Extract frozen-CNN embeddings for train/val/test splits, per CNN seed.

Writes v2/results/embeddings/cnn_seed{S}/{split}_embeddings.npy and _labels.npy
(labels in {0,1}). Embeddings are float64 and L2-normalized per row, matching the
v1 convention so the encoder's amplitude interpretation holds.

    uv run python -m v2.experiments.extract_embeddings_v2 --seed 0
    uv run python -m v2.experiments.extract_embeddings_v2 --seed 0 --splits val test --limit 32768
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.classical.cnn import PCamCNN
from src.data.pcam_loader import get_pcam_dataset
from src.data.transforms import get_eval_transforms
from src.utils.paths import load_paths
from v2.core.io import REPO_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def emb_dir(seed: int) -> str:
    d = os.path.join(REPO_ROOT, "v2", "results", "embeddings", f"cnn_seed{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def checkpoint_path(seed: int) -> str:
    return os.path.join(REPO_ROOT, "v2", "results", "cnn", f"cnn_seed{seed}", "pcam_cnn_best.pt")


@torch.no_grad()
def extract_split(model, data_root, split, limit=None, batch_size=256):
    ds = get_pcam_dataset(data_root, split, get_eval_transforms())
    if limit is not None and limit < len(ds):
        ds = Subset(ds, range(limit))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=6, pin_memory=True)
    from tqdm import tqdm

    embeds, labels = [], []
    for x, y in tqdm(loader, desc=f"embed {split}", leave=False):
        z = model(x.to(DEVICE), return_embedding=True).to(torch.float64)
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        embeds.append(z.cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
    X = np.vstack(embeds).astype(np.float64)
    y = np.concatenate(labels).astype(np.int64)
    assert not np.any(np.linalg.norm(X, axis=1) == 0), "zero embedding encountered"
    return X, y


def main(seed: int, splits, limit):
    _, PATHS = load_paths()
    ckpt = checkpoint_path(seed)
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}; run train_cnn_v2 --seed {seed}"
    model = PCamCNN(embedding_dim=32).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    out = emb_dir(seed)
    for split in splits:
        X, y = extract_split(model, PATHS["dataset"], split, limit=limit)
        np.save(os.path.join(out, f"{split}_embeddings.npy"), X)
        np.save(os.path.join(out, f"{split}_labels.npy"), y)
        print(f"[seed {seed}] {split}: X{X.shape} pos_frac={y.mean():.3f} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.seed, args.splits, args.limit)
