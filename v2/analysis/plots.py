"""Plots for v2: shot sweep w/ theory overlay, noise curves, capacity curve.

    uv run python -m v2.analysis.plots shot   v2/results/shot_sweep/shot_sweep.json
    uv run python -m v2.analysis.plots noise  v2/results/noise_depolarizing/noise_depolarizing.json
    uv run python -m v2.analysis.plots capacity v2/results/capacity_sweep
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_shot(json_path):
    with open(json_path) as f:
        d = json.load(f)
    shots = d["shots"]
    out_dir = os.path.dirname(json_path)
    for name, m in d["models"].items():
        c = m["curves"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(shots, c["empirical_mean"], yerr=c["empirical_std"], marker="o",
                    label="empirical (sampled)", capsize=3)
        ax.plot(shots, c["predicted"], "--", label="binomial prediction")
        ax.axhline(m["exact_backend_acc"], color="gray", ls=":", label="exact backend")
        ax.set_xscale("log")
        ax.set_xlabel("shots"); ax.set_ylabel("accuracy")
        ax.set_title(f"Shot sweep: {name}"); ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"shot_sweep_{name}.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        print(f"wrote {p}")

        # Hoeffding error bound (separate panel)
        fig, ax = plt.subplots(figsize=(6, 4))
        emp_err = 1 - np.array(c["empirical_mean"])
        ax.plot(shots, emp_err, "o-", label="empirical error")
        ax.plot(shots, c["hoeffding_error"], "--", label="Hoeffding bound")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("shots"); ax.set_ylabel("error")
        ax.set_title(f"Hoeffding bound: {name}"); ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"hoeffding_{name}.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        print(f"wrote {p}")


def plot_noise(json_path):
    with open(json_path) as f:
        d = json.load(f)
    out_dir = os.path.dirname(json_path)
    for name, m in d["models"].items():
        analytic = m.get("analytic", {})
        if not analytic:
            continue
        ps = sorted(float(p) for p in analytic)
        fig, ax = plt.subplots(figsize=(6, 4))
        shots_keys = list(next(iter(analytic.values()))["acc_by_shots"].keys())
        for sh in shots_keys:
            accs = [analytic[str(p)]["acc_by_shots"][sh]["mean"] for p in ps]
            ax.plot(ps, accs, marker="o", label=f"{sh} shots")
        cv = m.get("circuit_validation", {})
        if cv:
            cps = sorted(float(p) for p in cv)
            ax.scatter(cps, [cv[str(p)]["circuit_acc"] for p in cps], color="k",
                       zorder=5, marker="x", s=80, label="circuit (aer)")
        ax.set_xscale("log")
        ax.set_xlabel("depolarizing p"); ax.set_ylabel("accuracy")
        ax.set_title(f"Depolarizing noise: {name}"); ax.legend(); ax.grid(alpha=0.3)
        p = os.path.join(out_dir, f"noise_{name}.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        print(f"wrote {p}")


def plot_capacity(exp_dir):
    summ = os.path.join(exp_dir, "summary.json")
    assert os.path.exists(summ), f"run aggregate first: {summ}"
    with open(summ) as f:
        table = json.load(f)["accuracy"]
    ks, means, stds = [], [], []
    knn = None
    for name, r in table.items():
        if name.startswith("iqc_fixed_K"):
            ks.append(int(name.split("K")[-1])); means.append(r["mean"]); stds.append(r["std"])
        elif name.startswith("knn"):
            knn = r["mean"]
    order = np.argsort(ks)
    ks = np.array(ks)[order]; means = np.array(means)[order]; stds = np.array(stds)[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, label="IQC (K prototypes)")
    if knn is not None:
        ax.axhline(knn, color="r", ls="--", label="kNN(5)")
    ax.set_xlabel("K prototypes / class"); ax.set_ylabel("test accuracy")
    ax.set_title("Capacity sweep"); ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(exp_dir, "capacity_sweep.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    kind, target = sys.argv[1], sys.argv[2]
    if kind == "shot":
        plot_shot(target)
    elif kind == "noise":
        plot_noise(target)
    elif kind == "capacity":
        plot_capacity(target)
    else:
        raise SystemExit(f"unknown plot kind {kind}")
