"""Noise sweep: analytic tier (full test) + aer circuit-validation tier (subset).

Depolarizing:
  * analytic — attenuate exact overlaps by (1-p1)^{n1}(1-p2)^{n2} (n1,n2 = transpiled
    1q/2q gate counts), then binomial-sample; run on the full eval split.
  * circuit  — real AerSimulator with a depolarizing NoiseModel, transpiled against
    the noisy basis, on a small subset; validates the analytic attenuation.

Thermal (T1/T2): circuit tier only (no clean analytic form).

    uv run python -m v2.experiments.run_noise_sweep --config v2/configs/noise_depolarizing.yaml
    uv run python -m v2.experiments.run_noise_sweep --config v2/configs/noise_thermal.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from v2.core.io import load_config, git_commit
from v2.core.seeding import make_rngs, spawn_int
from v2.core.data import load_split, stratified_subsample
from v2.core.models import build_model
from v2.core.backends import sampled_scores
from v2.core import noise as noisemod


def _fit_iqc(cfg):
    d = cfg["data"]
    cnn = int(d["cnn"])
    Xtr_full, ytr_full = load_split(cnn, d["fit_split"])
    Xte, yte = load_split(cnn, d["eval_split"])
    rng0 = make_rngs(cfg["seeds"][0])
    Xtr, ytr = stratified_subsample(Xtr_full, ytr_full, d.get("fit_samples"), rng0["subsample"])
    models = {}
    for spec in cfg["models"]:
        m = build_model(spec, cfg["seeds"][0], rng_stream=rng0["stream"]).fit(Xtr, ytr)
        models[spec["name"]] = m
    return models, Xte, yte


def run_depolarizing(cfg):
    models, Xte, yte = _fit_iqc(cfg)
    ncfg = cfg["noise"]
    p_list = ncfg["p"]
    p2_factor = ncfg.get("p2_factor", 10)
    shots_list = cfg["shots"]
    basis = list(noisemod.BASIS_1Q) + list(noisemod.BASIS_2Q)

    out = {"experiment": cfg["name"], "git_commit": git_commit(), "config": cfg, "models": {}}
    for name, model in models.items():
        s_exact = model.decision_scores(Xte)
        chi = model.effective_chi
        gates = noisemod.transpiled_depth(chi, Xte[0], basis)
        n1 = sum(v for k, v in gates["ops"].items() if k in noisemod.BASIS_1Q)
        n2 = gates["num_2q"]

        analytic = {}
        for p in p_list:
            p2 = min(p * p2_factor, 1.0)
            atten = (1 - p) ** n1 * (1 - p2) ** n2
            row = {}
            for shots in shots_list:
                emp = []
                for seed in cfg["seeds"]:
                    rng = make_rngs(seed)["shots"]
                    s_hat = sampled_scores(s_exact, shots, rng, attenuation=atten)
                    emp.append(np.mean((s_hat >= 0).astype(int) == yte))
                row[str(shots)] = {"mean": float(np.mean(emp)),
                                   "std": float(np.std(emp, ddof=1) if len(emp) > 1 else 0.0)}
            analytic[str(p)] = {"attenuation": float(atten), "acc_by_shots": row}

        # circuit validation tier
        circ = {}
        cv = cfg.get("circuit_validation", {})
        if cv:
            n_sub = cv.get("eval_samples", 500)
            rng = make_rngs(cfg["seeds"][0])["subsample"]
            Xs, ys = stratified_subsample(Xte, yte, n_sub, rng)
            shots_c = shots_list[-1]
            for p in cv.get("p", []):
                p2 = min(p * p2_factor, 1.0)
                nm = noisemod.depolarizing_noise(p, p2)
                from qiskit_aer import AerSimulator
                from src.IQL.backends.hardware_native import HardwareNativeBackend
                sim = AerSimulator(noise_model=nm)
                hb = HardwareNativeBackend(backend=sim, shots=shots_c,
                                           seed_simulator=spawn_int(make_rngs(cfg["seeds"][0])["sim"]),
                                           basis_gates=nm.basis_gates)
                s_hat = hb.score_batch(chi, Xs)
                acc = float(np.mean((s_hat >= 0).astype(int) == ys))
                # analytic prediction at same p, same shots, same subset
                atten = (1 - p) ** n1 * (1 - p2) ** n2
                s_pred = sampled_scores(model.decision_scores(Xs), shots_c,
                                        make_rngs(cfg["seeds"][0])["shots"], attenuation=atten)
                acc_pred = float(np.mean((s_pred >= 0).astype(int) == ys))
                circ[str(p)] = {"circuit_acc": acc, "analytic_acc": acc_pred,
                                "n_sub": int(len(ys)), "shots": shots_c}
                print(f"[noise:circuit] {name} p={p}: circuit={acc:.4f} analytic={acc_pred:.4f}")

        out["models"][name] = {"gate_counts": {"n1": int(n1), "n2": int(n2), **gates},
                               "exact_acc": float(np.mean((s_exact >= 0).astype(int) == yte)),
                               "analytic": analytic, "circuit_validation": circ}
        print(f"[noise:analytic] {name}: n1={n1} n2={n2} depth={gates['depth']}")

    return out


def run_thermal(cfg):
    models, Xte, yte = _fit_iqc(cfg)
    ncfg = cfg["noise"]
    gate_ns = ncfg["gate_ns"]
    n_sub = cfg["data"].get("eval_samples", 500)
    shots = cfg["shots"][-1] if isinstance(cfg["shots"], list) else cfg["shots"]
    from qiskit_aer import AerSimulator
    from src.IQL.backends.hardware_native import HardwareNativeBackend

    out = {"experiment": cfg["name"], "git_commit": git_commit(), "config": cfg, "models": {}}
    for name, model in models.items():
        chi = model.effective_chi
        rng = make_rngs(cfg["seeds"][0])["subsample"]
        Xs, ys = stratified_subsample(Xte, yte, n_sub, rng)
        points = {}
        for pt in ncfg["points"]:
            nm = noisemod.thermal_noise(pt["T1_us"], pt["T2_us"], gate_ns)
            sim = AerSimulator(noise_model=nm)
            hb = HardwareNativeBackend(backend=sim, shots=shots,
                                       seed_simulator=spawn_int(make_rngs(cfg["seeds"][0])["sim"]),
                                       basis_gates=nm.basis_gates)
            s_hat = hb.score_batch(chi, Xs)
            acc = float(np.mean((s_hat >= 0).astype(int) == ys))
            points[f"T1_{pt['T1_us']}_T2_{pt['T2_us']}"] = {"acc": acc, "n_sub": int(len(ys))}
            print(f"[thermal] {name} T1={pt['T1_us']} T2={pt['T2_us']}: acc={acc:.4f}")
        out["models"][name] = {"points": points}
    return out


def run(config_path):
    cfg = load_config(config_path)
    kind = cfg["noise"]["kind"]
    out = run_depolarizing(cfg) if kind == "depolarizing" else run_thermal(cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    fn = os.path.join(cfg["output_dir"], f"noise_{kind}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {fn}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)
