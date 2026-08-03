"""Generic config-driven experiment driver (main comparison, capacity, regimes).

For each seed:
  * resolve the CNN embeddings (per_seed or pinned),
  * fit each model on a seeded stratified subsample of the fit split,
  * evaluate on the eval split (per-model caps allowed),
  * write results.json + per-example prediction / margin sidecars.

A `params` value that is a list expands into one model per element (used by the
capacity sweep over K). Everything is exact-backend here; shot / noise sweeps have
their own runners.

    uv run python -m v2.experiments.run_experiment --config v2/configs/main_comparison.yaml
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import accuracy_score

from v2.core.io import load_config, seed_dir, write_results, save_predictions, save_margins, git_commit
from v2.core.seeding import make_rngs
from v2.core.data import load_split, stratified_subsample
from v2.core.models import build_model


def _expand_models(models):
    """Expand any list-valued param into separate model specs (e.g. K sweep)."""
    out = []
    for spec in models:
        params = spec.get("params", {}) or {}
        list_keys = [k for k, v in params.items() if isinstance(v, list)]
        if not list_keys:
            out.append((spec["name"], spec))
            continue
        assert len(list_keys) == 1, f"only one list-valued param supported, got {list_keys}"
        key = list_keys[0]
        for val in params[key]:
            new_params = dict(params)
            new_params[key] = val
            tag = f"{spec['name']}_{key}{val}"
            new_spec = dict(spec)
            new_spec["params"] = new_params
            out.append((tag, new_spec))
    return out


def _cnn_seed(cfg, seed):
    c = cfg["data"]["cnn"]
    return seed if c == "per_seed" else int(c)


def run_seed(cfg, seed):
    rngs = make_rngs(seed)
    cnn_seed = _cnn_seed(cfg, seed)
    d = cfg["data"]

    Xtr_full, ytr_full = load_split(cnn_seed, d["fit_split"])
    Xte_full, yte_full = load_split(cnn_seed, d["eval_split"])

    run_dir = seed_dir(cfg, seed)
    results = {
        "experiment": cfg["name"],
        "seed": seed,
        "git_commit": git_commit(),
        "config": cfg,
        "data": {"cnn_seed": cnn_seed, "fit_split": d["fit_split"], "eval_split": d["eval_split"]},
        "models": {},
    }

    # Cache subsamples by cap so every model with the same cap sees identical data,
    # independent of model order (fairness + determinism).
    fit_cache, eval_cache = {}, {}

    def get_fit(n):
        if n not in fit_cache:
            r = np.random.default_rng([seed, 1, n if n is not None else -1])
            fit_cache[n] = stratified_subsample(Xtr_full, ytr_full, n, r)
        return fit_cache[n]

    def get_eval(n):
        if n not in eval_cache:
            if n is None or n >= len(Xte_full):
                eval_cache[n] = (Xte_full, yte_full)
            else:
                r = np.random.default_rng([seed, 2, n])
                eval_cache[n] = stratified_subsample(Xte_full, yte_full, n, r)
        return eval_cache[n]

    for tag, spec in _expand_models(cfg["models"]):
        fit_n = spec.get("fit_samples", d.get("fit_samples"))
        eval_n = spec.get("eval_samples", d.get("eval_samples"))
        Xtr, ytr = get_fit(fit_n)
        Xte, yte = get_eval(eval_n)

        t0 = time.time()
        model = build_model(spec, seed, rng_stream=rngs["stream"])
        model.fit(Xtr, ytr)
        fit_t = time.time() - t0

        t1 = time.time()
        preds = np.asarray(model.predict(Xte)).astype(int)
        eval_t = time.time() - t1
        acc = float(accuracy_score(yte, preds))

        pfile = save_predictions(run_dir, tag, preds)
        entry = {
            "params": spec.get("params", {}),
            "backend": "exact",
            "accuracy": acc,
            "n_correct": int((preds == yte).sum()),
            "n_eval": int(len(yte)),
            "fit_samples": int(len(Xtr)),
            "fit_time_s": round(fit_t, 3),
            "eval_time_s": round(eval_t, 3),
            "predictions_file": pfile,
            "extra": {},
        }
        if getattr(model, "is_iqc", False):
            try:
                margins = np.abs(model.decision_scores(Xte))
                entry["margins_file"] = save_margins(run_dir, tag, margins)
            except Exception:
                pass
            if getattr(model, "n_memories", None) is not None:
                entry["extra"]["n_memories"] = int(model.n_memories)
        results["models"][tag] = entry
        print(f"[{cfg['name']} seed {seed}] {tag}: acc={acc:.4f} "
              f"(fit {fit_t:.1f}s eval {eval_t:.1f}s, n_eval={len(yte)})")

    # save eval labels once for downstream McNemar
    np.save(f"{run_dir}/eval_labels.npy", yte_full.astype(np.int8))
    write_results(run_dir, results)
    return results


def main(config_path):
    cfg = load_config(config_path)
    for seed in cfg["seeds"]:
        run_seed(cfg, seed)
    print(f"done: {cfg['name']} over seeds {cfg['seeds']} -> {cfg['output_dir']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
