"""Unified model factory for v2.

Every wrapper exposes:
    fit(Xtr, ytr_binary) -> self
    predict(Xte)         -> np.ndarray of labels in {0, 1}
    decision_scores(Xte) -> np.ndarray of reals; higher => more class-1

IQC-family wrappers additionally expose `effective_chi`, the single unit prototype
whose overlap sign reproduces the frozen classifier (proven in math_foundations.md
and the equivalence audit). This is what the shot / noise sweeps sample: one
Hadamard test against effective_chi, decision_scores(X) = X @ effective_chi in
[-1, 1] = the estimable overlap.

Fitting always uses the exact backend (fast, deterministic); the sampled/circuit
backends are applied only at inference by the sweep runners.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.IQL.backends.exact import ExactBackend
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.models.static_isdo_model import StaticISDOModel
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.models.adaptive_memory_model import AdaptiveMemoryModel
from src.IQL.regimes.regime2_online import OnlinePerceptron
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning


def _polar(y):
    return (np.asarray(y) * 2 - 1).astype(int)


def _orient(weight, X, y_bin):
    """Pick global sign so sign(o*weight . x >= 0) best matches class-1 labels."""
    s = X @ weight
    acc_pos = np.mean((s >= 0).astype(int) == y_bin)
    return weight if acc_pos >= 0.5 else -weight


# --------------------------------------------------------------------------- IQC


class _IQCLinear:
    """Frozen single-hyperplane IQC (static / fixed / regime2)."""

    is_iqc = True

    def __init__(self, kind, params, seed, rng_stream=None):
        self.kind = kind
        self.params = params or {}
        self.seed = seed
        self.rng_stream = rng_stream
        self.effective_chi = None
        self.n_memories = None

    def _raw_weight(self, X, y_bin):
        if self.kind == "iqc_static":
            m = StaticISDOModel(K=self.params.get("K", 1), seed=self.seed).fit(X, y_bin)
            # StaticISDOClassifier: class1 iff chi.x < 0  => class-1 weight is -chi
            self.n_memories = 2 * self.params.get("K", 1)
            return -np.real(m.classifier.chi)
        if self.kind == "iqc_fixed":
            m = FixedMemoryIQC(K=self.params.get("K", 3), eta=self.params.get("eta", 0.1),
                               backend=ExactBackend(), seed=self.seed).fit(X, _polar(y_bin))
            mem = m.classifier.memory_bank.class_states
            self.n_memories = len(mem)
            W = np.zeros_like(np.real(mem[0].vector))
            for w, cs in zip(m.classifier.weights, mem):
                W = W + w * np.real(cs.vector)
            return W
        if self.kind == "iqc_regime2":
            # cold-start chi from the class-difference centroid (Regime 1 -> 2)
            yp = _polar(y_bin)
            chi0 = X[yp == 1].mean(0) - X[yp == -1].mean(0)
            cs = ClassState(chi0.astype(np.complex128), label=1, backend=ExactBackend())
            order = (self.rng_stream.permutation(len(X)) if self.rng_stream is not None
                     else np.arange(len(X)))
            perc = OnlinePerceptron(cs, eta=self.params.get("eta", 0.1))
            perc.fit(X[order], yp[order])
            self.n_memories = 1
            return np.real(cs.vector)
        raise ValueError(self.kind)

    def fit(self, X, y_bin):
        w = self._raw_weight(X, y_bin)
        w = _orient(w, X, y_bin)
        self.effective_chi = w / np.linalg.norm(w)
        return self

    def decision_scores(self, X):
        return X @ self.effective_chi

    def predict(self, X):
        return (self.decision_scores(X) >= 0).astype(int)


class _IQCAdaptive:
    """Regime 4A+4B adaptive multi-memory model (piecewise-linear inference)."""

    is_iqc = True
    effective_chi = None  # genuinely nonlinear; not sampled in shot sweep

    def __init__(self, params, seed, rng_stream=None):
        self.params = params or {}
        self.seed = seed
        self.rng_stream = rng_stream
        self.model = None
        self.n_memories = None
        self._flip = 1

    def fit(self, X, y_bin):
        yp = _polar(y_bin)
        rng = self.rng_stream if self.rng_stream is not None else np.random.default_rng(self.seed)
        backend = ExactBackend()
        class_states = []
        for cls in [-1, +1]:
            idx = rng.choice(np.where(yp == cls)[0])
            chi = X[idx].astype(np.complex128)
            class_states.append(ClassState(chi / np.linalg.norm(chi), label=cls, backend=backend))
        bank = MemoryBank(class_states)
        learner = Regime4ASpawn(memory_bank=bank, eta=self.params.get("eta", 0.1), backend=backend,
                                delta_cover=self.params.get("delta_cover", 0.2),
                                spawn_cooldown=self.params.get("spawn_cooldown", 100),
                                min_polarized_per_class=1)
        pruner = Regime4BPruning(memory_bank=bank, tau_harm=self.params.get("tau_harm", -0.15),
                                 min_age=200, min_per_class=1, prune_interval=200)
        self.model = AdaptiveMemoryModel(bank, learner, pruner, tau_responsible=0.1, beta=0.98)
        order = rng.permutation(len(X))
        self.model.fit(X[order], yp[order])
        self.model.consolidate(X[order], yp[order],
                               epochs=self.params.get("consolidate_epochs", 5), eta_scale=0.3)
        self.n_memories = len(bank.class_states)
        # orient predictions to {0,1}
        preds_polar = np.array(self.model.predict(X))
        if np.mean(((preds_polar + 1) // 2) == y_bin) < 0.5:
            self._flip = -1
        return self

    def decision_scores(self, X):
        scores = np.array([self.model.memory_bank.winner(x)[1] for x in X])
        return self._flip * scores

    def predict(self, X):
        return (self.decision_scores(X) >= 0).astype(int)


# ----------------------------------------------------------------------- classic


class _Sklearn:
    is_iqc = False

    def __init__(self, kind, params, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.neighbors import KNeighborsClassifier

        p = params or {}
        if kind == "logreg":
            self.clf = LogisticRegression(C=p.get("C", 1.0), max_iter=1000, random_state=seed)
        elif kind == "linsvm":
            self.clf = LinearSVC(C=p.get("C", 1.0), random_state=seed)
        elif kind == "knn":
            self.clf = KNeighborsClassifier(n_neighbors=p.get("k", 5))
        else:
            raise ValueError(kind)
        self.kind = kind

    def fit(self, X, y_bin):
        self.clf.fit(X, y_bin)
        return self

    def decision_scores(self, X):
        if hasattr(self.clf, "decision_function"):
            return self.clf.decision_function(X)
        proba = self.clf.predict_proba(X)
        return proba[:, 1] - proba[:, 0]

    def predict(self, X):
        return self.clf.predict(X).astype(int)


# -------------------------------------------------------------------- factory


def build_model(spec: dict, seed: int, rng_stream: Optional[np.random.Generator] = None):
    name = spec["name"]
    params = spec.get("params", {})
    if name in ("iqc_static", "iqc_fixed", "iqc_regime2"):
        return _IQCLinear(name, params, seed, rng_stream)
    if name == "iqc_adaptive":
        return _IQCAdaptive(params, seed, rng_stream)
    if name in ("logreg", "linsvm", "knn"):
        return _Sklearn(name, params, seed)
    if name == "qsvm":
        from v2.core.quantum_baselines import QSVMWrapper

        return QSVMWrapper(params, seed)
    if name == "vqc":
        from v2.core.quantum_baselines import VQCWrapper

        return VQCWrapper(params, seed)
    raise ValueError(f"unknown model {name}")
