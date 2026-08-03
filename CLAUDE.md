# Interference Quantum Classifier (IQC)

## Environment & commands

`uv`-managed Python (>=3.10, `.python-version` pins the venv) — `uv sync` to install,
`uv add <pkg>` to add deps, `uv run <script>.py` to execute. No test suite or linter
configured (one stray `test.py` at repo root, unrelated to pytest). Key deps: `qiskit`
1.3.2 + `qiskit-aer-gpu` 0.15.1 + `qiskit-machine-learning` (measurement-based quantum
baselines), `torch`/`torchvision` (CNN feature extractor), `qutip`, `scikit-learn`.

All data/output paths are driven by `configs/paths.yaml` via `src/utils/paths.py:load_paths()`
(`base_root` joined with each relative path). `base_root` was stale (pointed at an old
home-dir checkout) and has been fixed to `/data/tarakesh/Interference-quantum-classifier`.

Dataset: PatchCamelyon (PCam), downloaded via `torchvision.datasets.PCAM` through
`src/data/pcam_loader.py:get_pcam_dataset()`, lives at `dataset/pcam/*.h5`
(train 262144 / val 32768 / test 32768 patches, ~8.5G after dropping the `.h5.gz`
originals — safe to delete, torchvision doesn't need them once extracted).

Pipeline order to reproduce from scratch:
```
uv run src/training/classical/train_cnn.py            # trains PCamCNN, saves checkpoint
uv run src/training/classical/extract_embeddings.py    # frozen CNN -> 32D embeddings, results/embeddings/
uv run src/training/classical/make_embedding_split.py  # train/test index splits (if not cached)
uv run src/generate_final_comparison.py                # classical + StaticISDO + FixedMemoryIQC -> results/final_comparison_results.json
uv run src/evaluate_iqc_vs_classical.py                # classical vs AdaptiveMemoryModel (Regime-4A/4B)
uv run src/evaluate_capacity_sweep_quantum_vs_knn.py   # K sweep vs KNN -> results/figures/
uv run src/evaluate_all_qmls.py                        # static/fixed/adaptive IQC side by side
uv run src/quantum/train_test_qsvm_amp_encode.py       # QSVC measurement-based baseline
uv run src/quantum/train_test_vqc_amp_encode.py        # VQC measurement-based baseline
```

## What this does

Tests whether a **measurement-free** quantum similarity classifier can match classical
and measurement-based-quantum baselines on PatchCamelyon histopathology patches, while
running orders of magnitude faster (statevector-exact, no shot sampling). A frozen CNN
(`src/classical/cnn.py`) is only a feature extractor to 32-D embeddings; every subsequent
"learning" step manipulates quantum-state amplitudes directly in classical vector space
— no gate-parameter training, no barren plateaus.

The decision primitive is **ISDO** (Interference-Sign Decision Observable):
`O(ψ;χ) = Re⟨χ|ψ⟩`, `ŷ = sign(O)` — linear (not quadratic/fidelity like SWAP-test),
phase-sensitive, and realizable via a Hadamard-test-style ancilla circuit using a
transition unitary `U_χψ = U_χ·U_ψ†` (`src/IQL/backends/transition.py:54-79`, backend
"ISDO-B′"), so only that one circuit ever touches actual quantum simulation.

Current headline numbers (README.md, 5000 PCam val samples): LogReg 0.909, k-NN(5) 0.926,
SWAP-test(1024 shots) 0.875, IQC single-prototype 0.876, IQC K=3 0.886 — i.e.
representation strategy (how prototypes aggregate), not measurement precision, is the
bottleneck.

## Architecture

1. **CNN embeddings** (`src/training/classical/`): `train_cnn.py` trains `PCamCNN`
   (`src/classical/cnn.py`); `extract_embeddings.py` runs it frozen over PCam
   (`src/data/pcam_loader.py`) and caches `val_embeddings.npy`/`val_labels.npy`/
   `split_*_idx.npy` under `results/embeddings/`. Everything downstream reads these via
   `src/utils/load_data.py`.

2. **State encoding** (`src/IQL/encoding/embedding_to_state.py`): L2-normalizes an
   embedding into an amplitude-encoded statevector `|ψ⟩`.

3. **Backends** (`src/IQL/backends/`) implement `InterferenceBackend.score(chi, psi)`:
   - `exact.py` — direct `Re⟨χ|ψ⟩` via numpy, no circuit (used for all the sweeps above).
   - `transition.py` — the physically realizable ISDO-B′ Hadamard-test circuit.
   - `hardware_native.py` — shot-based sampling version.

4. **Learning rules** (`src/IQL/learning/`): `update.py` implements the Regime-2
   quantum-perceptron rule — on misclassification, `χ ← normalize(χ + η·y·ψ)`.
   "Regimes" describe *how χ (or multiple χ's) get built/updated*; the circuit and
   observable never change across regimes:
   - **Regime 1 (static)** — one-shot prototype aggregation `χ = normalize(Σφ⁺ − Σφ⁻)`,
     no training loop (`StaticISDOClassifier`/`StaticISDOModel`).
   - **Regime 2 (online)** — single-χ quantum perceptron (`src/IQL/regimes/regime2_online.py`).
   - **Regime 3A (winner-take-all)** — K prototypes/class, only the highest-|score|
     memory updates per step (`regime3a_wta.py`, used by `FixedMemoryIQC`, fixed K, no
     growth/pruning).
   - **Regime 3B (responsible)** — variant with harm/responsibility bookkeeping feeding
     pruning (`src/training/protocol_fixed_regime3b_responsible/`).
   - **Regime 4A (spawn)** — when the winner covers poorly (`|score| < delta_cover`) and
     misclassifies (cooldown permitting), orthogonalize the residual and spawn a new
     memory state (`regime4a_spawn.py:add_memory`).
   - **Regime 4B (pruning)** — periodically drops memories old enough (`age >= min_age`)
     with harmful EMA score (`harm_ema < tau_harm`), respecting a per-class floor.
   - `AdaptiveMemoryModel` (`src/IQL/models/adaptive_memory_model.py`) orchestrates
     4A+4B each step plus a post-hoc consolidation phase that freezes structure and
     refines existing memories via Regime-3A updates.
   - Progression: Regime 1 → cold-start init for Regime 2 → generalizes to Regime 3
     (multi-memory) → made adaptive in Regime 4A/4B. Matches
     `status_reports/2,1,1_regimes.md` closely — that doc is the design reference if
     code behavior is unclear.

5. **Baselines**: classical (sklearn LogReg/SVC/KNN, inline in the `evaluate_*.py`
   scripts); measurement-based quantum — `src/quantum/train_test_qsvm_amp_encode.py`
   (QSVC + FidelityQuantumKernel) and `train_test_vqc_amp_encode.py` (VQC, COBYLA).

6. **Entry points**: `src/generate_final_comparison.py`, `evaluate_iqc_vs_classical.py`,
   `evaluate_all_qmls.py`, `evaluate_capacity_sweep_quantum_vs_knn.py` are the real
   comparison scripts. `src/quick_accuracy_test.py` re-evaluates a pickled
   `FixedMemoryIQC` model.

### Known landmines

- **`src/run_with_shorts_IQL_QSVM_VQC.py`** is the only IQL-vs-QSVM-vs-VQC shots-sweep
  script and **does nothing when run directly**: `main()` is commented out at the bottom
  (~line 165) and the `__main__` block just prints a hardcoded historical log string from
  a past run. Uncomment `main()` to actually execute the sweep (lines ~24-163).
- `src/data/pcam_loader.py:4` — the `data_dir` default arg is still the old
  `/home/tarakesh/Work/Repo/measurement-free-quantum-classifier/dataset` path. Harmless
  today (every call site passes `PATHS["dataset"]` explicitly) but a trap if ever called
  bare from a notebook/new script.
- Several scripts embed old absolute paths from a previous machine inside
  docstrings/printed strings that just echo historical run output (e.g.
  `evaluate_capacity_sweep_quantum_vs_knn.py:233`, `train_test_vqc_amp_encode.py:171-180`)
  — inert, not a functional bug, but easy to mistake for a live hardcoded path.
- Repo-wide convention: several `evaluate_*`/`protocol_static` scripts append trailing
  docstrings with last-known-good numbers after the executable code (e.g.
  `"""ISDO Accuracy (test): 0.8840"""`) — this is intentional inline documentation of
  prior run results, not dead/commented-out logic.
- `StaticISDOClassifier.predict_one` (`src/IQL/baselines/static_isdo_classifier.py:26-27`)
  returns `1 if score < 0 else 0` — binary labels, inverted-looking vs. the polar
  `{-1,+1}` convention used by `AdaptiveMemoryModel.predict`/`WeightedVoteClassifier`.
  Internally consistent given how `chi` is built there, but no comment flags the sign
  flip — worth double-checking if extending regimes to reuse this classifier.
- `StaticISDOClassifier` prototype dict only has keys `0`/`1` — passing a prototype with
  `label=None` will `KeyError` (`static_isdo_classifier.py:21`); not currently triggered.
- Filename typo: `src/training/protocol_fixed_regime3b_responsible/test_regime3b_egime3b_responsible.py`.
