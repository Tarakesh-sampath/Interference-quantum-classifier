"""Aggregate per-seed results.json into a mean±std table + McNemar matrix.

    uv run python -m v2.analysis.aggregate --exp_dir v2/results/main_comparison
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from v2.core.stats import aggregate, mcnemar_matrix


def load_runs(exp_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "seed_*", "results.json"))):
        with open(path) as f:
            runs.append((os.path.dirname(path), json.load(f)))
    assert runs, f"no results.json under {exp_dir}/seed_*"
    return runs


def build_table(runs):
    accs = defaultdict(list)
    extras = defaultdict(list)
    for _, r in runs:
        for name, m in r["models"].items():
            accs[name].append(m["accuracy"])
            if m.get("extra", {}).get("n_memories") is not None:
                extras[name].append(m["extra"]["n_memories"])
    table = {}
    for name, vals in accs.items():
        row = aggregate(vals)
        if extras.get(name):
            row["n_memories_mean"] = float(np.mean(extras[name]))
        table[name] = row
    return table


def per_seed_mcnemar(runs):
    """Median McNemar p-value per model pair across seeds."""
    pair_p = defaultdict(list)
    for run_dir, r in runs:
        labels_path = os.path.join(run_dir, "eval_labels.npy")
        if not os.path.exists(labels_path):
            continue
        y = np.load(labels_path).astype(int)
        preds = {}
        for name, m in r["models"].items():
            pf = os.path.join(run_dir, m.get("predictions_file", ""))
            if os.path.exists(pf):
                p = np.load(pf).astype(int)
                if len(p) == len(y):
                    preds[name] = p
        if len(preds) < 2:
            continue
        mat = mcnemar_matrix(preds, y)
        for a in mat:
            for b, pv in mat[a].items():
                if a < b:
                    pair_p[(a, b)].append(pv)
    return {f"{a} vs {b}": float(np.median(v)) for (a, b), v in pair_p.items()}


def to_markdown(table):
    lines = ["| model | acc mean | std | min | max | n | n_mem |",
             "|---|---|---|---|---|---|---|"]
    for name in sorted(table, key=lambda k: -table[k]["mean"]):
        r = table[name]
        nm = f"{r.get('n_memories_mean', ''):.1f}" if "n_memories_mean" in r else ""
        lines.append(f"| {name} | {r['mean']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} | {r['n']} | {nm} |")
    return "\n".join(lines)


def main(exp_dir):
    runs = load_runs(exp_dir)
    table = build_table(runs)
    md = to_markdown(table)
    mcn = per_seed_mcnemar(runs)

    out = {"experiment": os.path.basename(exp_dir.rstrip("/")),
           "n_seeds": len(runs), "accuracy": table, "mcnemar_median_p": mcn}
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(exp_dir, "summary.md"), "w") as f:
        f.write(f"# {out['experiment']} ({len(runs)} seeds)\n\n{md}\n\n")
        if mcn:
            f.write("## McNemar median p-values (paired, per-seed)\n\n")
            for k, v in sorted(mcn.items(), key=lambda kv: kv[1]):
                f.write(f"- {k}: p={v:.4g}\n")
    print(md)
    print(f"\nwrote {exp_dir}/summary.json and summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()
    main(args.exp_dir)
