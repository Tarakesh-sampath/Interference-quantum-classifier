"""Config loading, run-directory layout, and results serialization for v2.

Deliberately light: a few asserts instead of a schema framework. Every run dir
gets a full copy of its config and the current git commit so any number is
traceable back to exactly what produced it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_KEYS = ("name", "seeds", "data", "models", "output_dir")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"config {path} missing required keys: {missing}"
    assert isinstance(cfg["seeds"], list) and cfg["seeds"], "seeds must be a non-empty list"
    assert isinstance(cfg["models"], list) and cfg["models"], "models must be a non-empty list"
    cfg.setdefault("backend", {"kind": "exact"})
    cfg.setdefault("shots", None)
    cfg.setdefault("noise", None)
    # normalize output_dir to an absolute path under the repo
    if not os.path.isabs(cfg["output_dir"]):
        cfg["output_dir"] = os.path.join(REPO_ROOT, cfg["output_dir"])
    return cfg


def seed_dir(cfg: Dict[str, Any], seed: int) -> str:
    d = os.path.join(cfg["output_dir"], f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    return d


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_results(run_dir: str, results: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, "results.json")
    with open(path, "w") as f:
        json.dump(_jsonable(results), f, indent=2, sort_keys=True)
    return path


def save_predictions(run_dir: str, model_name: str, preds: np.ndarray) -> str:
    fname = f"preds_{model_name}.npy"
    np.save(os.path.join(run_dir, fname), np.asarray(preds, dtype=np.int8))
    return fname


def save_margins(run_dir: str, model_name: str, margins: np.ndarray) -> str:
    fname = f"margins_{model_name}.npy"
    np.save(os.path.join(run_dir, fname), np.asarray(margins, dtype=np.float64))
    return fname
