# IQC v2 — Standardized, Multi-Seed Experimental Rewrite + Math Foundations

## Context

The IQC paper (`paper_writeup/Tarakeshwaran_IQC_paper.pdf`) claims 88.07% on PCam with a
"quantum interference" classifier. Analysis of the code shows the decision function is
**classical**: embeddings are real/non-negative (ReLU head), so `Re⟨χ|ψ⟩` is exactly cosine
similarity to a signed class centroid — `exact.py` literally computes `np.real(np.vdot(chi,psi))`.
The genuinely quantum content is only the *execution story*: ⌈log₂d⌉-qubit storage and
constant-shot **sign** readout via the Hadamard test. v2 makes this honest and rigorous:
a math-foundations doc + empirical equivalence audit, and a redo of all experiments in a
`v2/` folder with multi-seed statistics, proper split hygiene, and real noise models.

**Defects in v1 that force the redo** (all verified in code):

1. **Transform bug** — `get_pcam_dataset(data_dir, split, download, transform)` but every
   caller (`train_cnn.py:79-80`, `extract_embeddings.py:25`) passes transforms as the 3rd
   *positional* arg → lands in `download`. The CNN was trained on un-normalized `[0,1]`
   tensors with **no augmentation and no Normalize(0.5,0.5)**. Forces a CNN retrain.
2. **No noise code exists** in the repo, yet the paper reports depolarizing-noise results
   (Table II, p=0.01). v2 must implement it for real (qiskit-aer NoiseModel).
3. **Split leakage** — all numbers come from the first 5000 samples of PCam *val*
   (deterministic slice), split 70/30; PCam *test* (32768) is never used.
4. **VQC shot-sweep bug** — `run_with_shorts_IQL_QSVM_VQC.py` reassigns `vqc_model.sampler`,
   but the underlying `SamplerQNN` keeps its own sampler reference → VQC accuracy constant
   (0.815) across all shot counts (paper Table III). Also `main()` is commented out.
5. **Hardcoded seed=42 everywhere** (`load_data`, `FixedMemoryIQC`, `StaticISDOModel`),
   unseeded `np.random.randn` fallback in `transition.py:40`, mutable default
   `backend=ExactBackend()` in `regime3a_wta.py`. Single-run results, no mean±std.

## Import strategy

v2 **imports from `src/` and patches `src/` minimally in place** (no copying — the audit must
test the actual paper code). `pyproject.toml` already builds `src` as a package; add `"v2*"`
to `[tool.setuptools.packages.find].include`. Run everything from repo root:
`uv run python -m v2.experiments.run_experiment --config v2/configs/<exp>.yaml`.

First implementation step: copy this plan into the repo as `v2/PLAN.md`.

## Directory layout

```
v2/
├── PLAN.md                       # this plan, checked in
├── core/
│   ├── seeding.py                # SeedSequence(seed).spawn() -> named sub-RNGs
│   │                             #   ("subsample","kmeans","stream","shots","sim")
│   ├── data.py                   # single data harness: load_embeddings(cnn_seed) ->
│   │                             #   {split: (X,y)}; stratified subsample(X,y,n,rng);
│   │                             #   explicit L2-normalize; NO hidden set_seed
│   ├── models.py                 # factory: config dict -> fitted model with unified
│   │                             #   predict(X) + raw_scores(X); absorbs the inverted
│   │                             #   label convention of StaticISDOClassifier in ONE place
│   ├── backends.py               # 3 eval modes: exact (S = X @ chi, vectorized) |
│   │                             #   sampled (analytic Hadamard test: P0=(1+s)/2,
│   │                             #   n0~Binomial(shots,P0), score=(2n0-shots)/shots) |
│   │                             #   circuit (batched HardwareNativeBackend + aer)
│   │                             #   + hoeffding_shots(margin, delta) + binomial flip-prob
│   ├── noise.py                  # depolarizing_noise(p1,p2), thermal_noise(T1,T2,gate_ns)
│   ├── stats.py                  # mean±std aggregation; McNemar (exact if b+c<25 else chi2)
│   └── io.py                     # yaml load/validate, run dirs, results.json, git-hash stamp
├── experiments/
│   ├── train_cnn_v2.py           # fixed transforms, --seed S
│   ├── extract_embeddings_v2.py  # train/val/test embeddings per CNN checkpoint
│   ├── run_experiment.py         # generic driver (main comparison, capacity, regimes)
│   ├── run_equivalence.py        # the quantumness audit
│   ├── run_shot_sweep.py         # binomial fast path + Hoeffding overlay
│   └── run_noise_sweep.py        # analytic tier + aer circuit validation tier
├── analysis/
│   ├── aggregate.py              # seed_*/results.json -> mean±std table + McNemar matrix
│   └── plots.py                  # shot curve w/ theory overlay, noise curves, K curve
├── configs/                      # equivalence / main_comparison / shot_sweep /
│                                 # noise_depolarizing / noise_thermal / capacity_sweep /
│                                 # adaptive_regimes .yaml
├── tests/
│   ├── test_equivalence.py       # exact IQC == signed-centroid classifier, 100% agreement
│   ├── test_seeding.py           # same seed -> bit-identical; different seeds differ
│   └── test_sampling.py          # binomial fast path ≡ noiseless aer circuit (statistical)
├── docs/math_foundations.md
└── results/                      # gitignored: <exp>/seed_<S>/results.json + .npy sidecars
```

## `docs/math_foundations.md` contents (written FIRST — gates all claims)

1. Setup: amplitude encoding Φ(x)=x/‖x‖, ISDO observable, Hadamard-test derivation
   P(0) = ½ + ½Re⟨χ|ψ⟩.
2. **Equivalence theorem**: real amplitude encoding ⇒ static IQC ≡ signed-centroid cosine
   classifier `sign(x·χ)`; FixedMemoryIQC ≡ WTA over KMeans prototypes (classical LVQ family).
3. **Shot-complexity bound**: ancilla estimator is Binomial ⇒ Hoeffding
   N ≈ log(1/δ)/(2·margin²) for correct sign — d-independent. This is the paper's real claim,
   now derived and tied to the empirical margin distribution.
4. **Noise propagation**: depolarizing ⇒ score × (1−p)^L (sign-preserving) ⇒ required shots
   × (1−p)^(−2L); L = transpiled circuit depth, logged from the real circuit.
5. Honest positioning: state prep costs O(d) gates (Plesch-Brukner) — the asterisk on any
   asymptotic advantage. Regime 2 = perceptron; Regimes 3/4 = online LVQ.

## Config schema (minimal YAML, one file per experiment)

```yaml
name: main_comparison
seeds: [0, 1, 2, 3, 4]
data:
  cnn: per_seed          # "per_seed" or int to pin one CNN
  fit_split: train       # fit_samples: 20000 (stratified; null = full)
  select_split: val      # hyperparam selection only (kNN k, LogReg C)
  eval_split: test       # eval_samples: null = full 32768
models:                  # per-model params + optional fit/eval caps
  - {name: iqc_static}
  - {name: iqc_fixed, params: {K: 3, eta: 0.1}}
  - {name: logreg} ...
  - {name: qsvm, fit_samples: 3500, eval_samples: 2000}
backend: {kind: exact}   # exact | sampled | circuit  (IQC-family only)
shots: null              # int or list -> sweep axis
noise: null              # {kind: depolarizing, p: ...} | {kind: thermal, T1_us, T2_us, gate_ns}
output_dir: v2/results/main_comparison
```

No cross-product engine — three explicit loops in three short runner scripts.

## Seed threading (minimal-diff patches to `src/`, backward-compatible defaults)

| Source | Mechanism |
|---|---|
| CNN train | `train_cnn_v2.py --seed S` → existing `set_seed(S)` |
| Subsampling | `rngs["subsample"]` in `core/data.py` |
| KMeans prototypes | patch `FixedMemoryIQC.__init__(..., seed=42)` and `StaticISDOModel` to pass through to `generate_prototypes(seed=...)` (`src/IQL/learning/prototype.py` **already accepts seed**) |
| Regime stream order | v2 shuffles (X,y) with `rngs["stream"]` before `fit` — no regime code change |
| Shot sampling | `rngs["shots"]` `np.random.Generator` into `core/backends.py` |
| Aer | patch `HardwareNativeBackend.__init__(..., seed_simulator=None)` |
| Gram-Schmidt fallback | patch `transition.py:40` to take optional rng (low priority) |

Also fix while patching: `regime3a_wta.py` mutable default `backend=ExactBackend()` → `None`;
`pcam_loader.py` signature reorder to `(data_dir, split, transform=None, download=True)` so
existing positional callers become *correct*. Total ≈ 6-line diff across ~5 files; v1 scripts
stay bit-identical (verify with `src/training/test_fixed_memory_iqc.py`).

## Experiment matrix (all final numbers on PCam test 32768; val = selection only)

| # | Config | Models / axis | Scale & cost |
|---|---|---|---|
| a | `equivalence` | iqc_static & iqc_fixed(K=3) vs classical NumPy replicas | v1 embeddings first (gate), then v2; **assert 100% agreement, |Δscore| < 1e-12**; seconds |
| b | `main_comparison` | IQC static/fixed K=3/adaptive(4a+4b), LogReg, LinSVM, kNN (fit 20k, eval 32768); QSVM & VQC capped fit 3500/eval 2000 (footnoted) | 5 seeds, per-seed CNN; ~2–3 h incl. retrains |
| c | `shot_sweep` | iqc_static + iqc_fixed; shots ∈ {1,2,4,…,4096}; overlay exact binomial flip-prob prediction + Hoeffding bound | 5 shot-seeds, pinned CNN, full test; minutes |
| d | `noise_*` | analytic tier: score×(1−p)^L, p ∈ {0.001…0.1} × shots {32,256,1024}, full test; circuit tier: batched aer + NoiseModel, 500-sample subset, 3 p values + 2–3 (T1,T2) points | minutes per tier (CPU aer — 6 qubits, GPU pointless) |
| e | `capacity_sweep` | iqc_fixed, K ∈ {1,2,3,5,7,11,13,17,19,23} (list already in `configs/paths.yaml`), kNN reference | 5 seeds, pinned CNN; minutes |
| f | `adaptive_regimes` | regime2 perceptron, regime3a WTA, 4a+4b AdaptiveMemoryModel + consolidate; track final prototype count | 5 stream-seeds, pinned CNN; minutes |

CNN policy: **retrain per seed for (b)** (transform bug forces ≥1 retrain anyway; ~1–2 h total
on the RTX PRO 6000; captures representation variance) — **pin cnn_seed0 for sweeps c–f**
(isolates the axis under study).

## Per-seed results + statistics

`v2/results/<exp>/seed_<S>/results.json`: experiment, seed, git_commit, full config copy,
per-model `{params, backend, shots, noise, accuracy, n_correct, fit/eval time, extra
(n_prototypes_final, transpiled_depth_L)}` + int8 `preds_*.npy` and float `margins_*.npy`
sidecars (needed for McNemar and the Hoeffding overlay).

`analysis/aggregate.py`: mean±std (min/max) per model → markdown + CSV; **McNemar per seed**
on paired predictions (exact binomial when discordant pairs < 25, else chi² with continuity
correction, via scipy) → per-pair p-value matrix + median across seeds. No pooling across seeds.

## Noise wiring (`core/noise.py`)

`depolarizing_error(p1,1)` on 1q basis gates + `depolarizing_error(p2,2)` on cx;
`thermal_relaxation_error(T1,T2,gate_time)` per gate. Inject via
`HardwareNativeBackend(backend=AerSimulator(noise_model=nm), shots, seed_simulator)`.
**Critical**: transpile with `basis_gates=nm.basis_gates` — otherwise `StatePreparation`
never decomposes into the noisy basis and errors silently never attach. Log transpiled
2q count + depth L from one representative circuit — the same L used in the analytic tier.
Batch all circuits into a single `backend.run(circuits)` (per-call transpile is the current
bottleneck in `hardware_native.py`).

## Phases & verification

- **P0 — patches + skeleton** (½ day): src/ seed patches, pcam_loader fix, pyproject include,
  `core/seeding.py`+`io.py`, config stubs, copy plan → `v2/PLAN.md`.
  ✓ v1 `test_fixed_memory_iqc.py` reproduces its prior number; `import v2.core.seeding` works.
- **P1 — math doc + equivalence gate**: `math_foundations.md` §1–5; `run_equivalence.py` +
  `tests/test_equivalence.py` on existing v1 embeddings.
  ✓ 100.0% agreement, |Δscore| < 1e-12 — **if not, stop, everything downstream is wrong**.
  ✓ `test_sampling.py`: binomial fast path vs noiseless circuit on 50 samples agree.
- **P2 — data pipeline**: `train_cnn_v2.py` ×5 seeds, `extract_embeddings_v2.py`
  (train/val/test per checkpoint), `core/data.py`.
  ✓ re-extraction under the *old buggy* transform reproduces v1 `val_embeddings.npy`
  (extraction fidelity regression); new CNN val acc ≥ old; no zero embeddings.
- **P3 — main comparison + cheap sweeps**: `run_experiment.py`; run (b), (e), (f).
  ✓ same seed twice → bit-identical results.json; different seeds differ; McNemar
  self-test p=1.0 on identical preds.
- **P4 — shot sweep** (c). ✓ empirical curve sits between exact binomial prediction (tight)
  and Hoeffding bound (loose); converges to exact-backend accuracy at high shots.
- **P5 — noise** (d). ✓ circuit tier matches analytic tier within binomial CI at 3 spot
  checks; p=0 noisy ≡ noiseless; fitted attenuation exponent ≈ logged L.
- **P6 — analysis**: `aggregate.py`, `plots.py`; fill empirical numbers into
  `math_foundations.md`. ✓ every headline number traceable to a results.json + git commit.

## Out of scope (explicitly)

Multi-class extension, real-hardware runs, rewriting the paper text (paper update happens
after v2 numbers exist).

## Key existing files touched/reused

- `src/IQL/models/fixed_memory_iqc.py`, `src/IQL/models/static_isdo_model.py` — seed param
- `src/IQL/learning/prototype.py` — already seedable; thread through
- `src/IQL/backends/hardware_native.py` — seed_simulator, basis-gate transpile, batched run
- `src/IQL/regimes/regime3a_wta.py` — mutable default fix
- `src/data/pcam_loader.py` — signature bug fix (forces CNN retrain)
- `src/utils/load_data.py` — untouched (defines the v1 baseline to regress against);
  replaced by `v2/core/data.py` for all v2 work
