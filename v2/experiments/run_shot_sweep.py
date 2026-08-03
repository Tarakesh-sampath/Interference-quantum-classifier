"""Shot sweep with theory overlay (Regime-1/3A single-hyperplane IQC).

Each frozen IQC model reduces to one prototype effective_chi; one Hadamard test per
sample estimates s = effective_chi . x. We sample that estimate at each shot count
(seeded binomial, the analytic-circuit-equivalent fast path) and record:

  * empirical accuracy of sign(estimate),
  * predicted accuracy from the exact per-sample binomial sign-flip probability,
  * Hoeffding error bound exp(-N m^2 / 2) averaged over the margin distribution.

    uv run python -m v2.experiments.run_shot_sweep --config v2/configs/shot_sweep.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from v2.core.io import load_config, git_commit
from v2.core.seeding import make_rngs
from v2.core.data import load_split, stratified_subsample
from v2.core.models import build_model
from v2.core.backends import sampled_scores, sign_flip_prob


def predicted_accuracy(exact_scores, y_bin, shots):
    """Tight binomial prediction of accuracy at `shots`."""
    fp = sign_flip_prob(exact_scores, shots)          # P(sampled sign flips)
    exact_pred = (exact_scores >= 0).astype(int)
    correct = (exact_pred == y_bin)
    p_correct = np.where(correct, 1.0 - fp, fp)
    return float(np.mean(p_correct))


def hoeffding_error(margins, shots):
    """Mean Hoeffding upper bound on per-sample sign error at `shots`."""
    return float(np.mean(np.exp(-shots * margins ** 2 / 2.0)))


def run(config_path):
    cfg = load_config(config_path)
    shots_list = cfg["shots"]
    assert isinstance(shots_list, list), "shot_sweep needs a list of shots"
    cnn = int(cfg["data"]["cnn"])
    d = cfg["data"]

    Xtr_full, ytr_full = load_split(cnn, d["fit_split"])
    Xte, yte = load_split(cnn, d["eval_split"])
    if d.get("eval_samples"):
        pass  # full test by default

    out = {"experiment": cfg["name"], "git_commit": git_commit(), "config": cfg,
           "shots": shots_list, "models": {}}

    for spec in cfg["models"]:
        name = spec["name"]
        curves = {"empirical_mean": [], "empirical_std": [], "predicted": [],
                  "hoeffding_error": []}
        # fit once on seed-0 subsample (pinned CNN); margins are model-defined
        rng0 = make_rngs(cfg["seeds"][0])
        Xtr, ytr = stratified_subsample(Xtr_full, ytr_full, d.get("fit_samples"), rng0["subsample"])
        model = build_model(spec, cfg["seeds"][0], rng_stream=rng0["stream"]).fit(Xtr, ytr)
        s_exact = model.decision_scores(Xte)          # in [-1,1]
        margins = np.abs(s_exact)
        exact_acc = float(np.mean((s_exact >= 0).astype(int) == yte))

        for shots in shots_list:
            emp = []
            for seed in cfg["seeds"]:
                rng = make_rngs(seed)["shots"]
                s_hat = sampled_scores(s_exact, shots, rng)
                emp.append(np.mean((s_hat >= 0).astype(int) == yte))
            curves["empirical_mean"].append(float(np.mean(emp)))
            curves["empirical_std"].append(float(np.std(emp, ddof=1) if len(emp) > 1 else 0.0))
            curves["predicted"].append(predicted_accuracy(s_exact, yte, shots))
            curves["hoeffding_error"].append(hoeffding_error(margins, shots))

        out["models"][name] = {
            "exact_backend_acc": exact_acc,
            "median_margin": float(np.median(margins)),
            "curves": curves,
        }
        print(f"[shot_sweep] {name}: exact_acc={exact_acc:.4f} "
              f"median_margin={np.median(margins):.3f}")
        for sh, em, pr in zip(shots_list, curves["empirical_mean"], curves["predicted"]):
            print(f"    shots={sh:5d}  empirical={em:.4f}  predicted={pr:.4f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "shot_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {cfg['output_dir']}/shot_sweep.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
