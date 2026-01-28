import numpy as np

from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning
from src.IQL.backends.exact import ExactBackend
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank


class AdaptiveIQC:
    """
    AdaptiveIQC: Self-Regulating Measurement-Free Quantum Classifier

    Combines:
    - Regime-3A: Winner-Take-All corrective learning (geometry)
    - Regime-4A: Coverage-based memory spawning (expansion)
    - Regime-4B: EMA-based responsible pruning (contraction)

    Memory topology and geometry adapt online.
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        *,
        # -------- Regime-3A (learner) --------
        eta: float = 0.1,
        alpha_correct: float = 0.0,
        alpha_wrong: float = 1.0,

        # -------- Regime-4A (spawn) ----------
        tau_spawn: float = 0.1,
        enable_spawn: bool = True,

        # -------- Regime-4B (prune) ----------
        tau_harm: float = -0.2,
        min_age: int = 200,
        min_per_class: int = 1,
        prune_interval: int = 200,
        tau_responsible: float = 0.1,
        harm_ema_beta: float = 0.98,
        enable_prune: bool = True,

        backend=None,
    ):
        self.backend = backend or ExactBackend()
        self.memory_bank = memory_bank

        # ---------------- Learner ----------------
        self.learner = WinnerTakeAll(
            memory_bank=self.memory_bank,
            eta=eta,
            backend=self.backend,
            alpha=alpha_correct,
            beta=alpha_wrong,
        )

        # ---------------- Spawner ----------------
        self.enable_spawn = enable_spawn
        if enable_spawn:
            self.spawner = Regime4ASpawn(
                memory_bank=self.memory_bank,
                tau_spawn=tau_spawn,
                backend=self.backend,
            )
        else:
            self.spawner = None

        # ---------------- Pruner ----------------
        self.enable_prune = enable_prune
        if enable_prune:
            self.pruner = Regime4BPruning(
                memory_bank=self.memory_bank,
                tau_harm=tau_harm,
                min_age=min_age,
                min_per_class=min_per_class,
                prune_interval=prune_interval,
            )
        else:
            self.pruner = None

        # -------- Regime-4B observation params --------
        self.tau_responsible = tau_responsible
        self.harm_ema_beta = harm_ema_beta

        # ---------------- Stats ----------------
        self.step_count = 0
        self.history = {
            "spawns": 0,
            "prunes": 0,
            "updates": 0,
            "memory_size": [],
        }

    # =================================================
    # Single training step
    # =================================================
    def step(self, psi, y_true):
        self.step_count += 1

        # -------- 1. Learning (Regime-3A) --------
        y_hat, _, updated = self.learner.step(psi, y_true)
        if updated:
            self.history["updates"] += 1

        # -------- 2. Update memory metadata --------
        self.memory_bank.increment_age()
        self.memory_bank.update_harm_ema(
            psi,
            tau_responsible=self.tau_responsible,
            beta=self.harm_ema_beta,
        )

        # -------- 3. Coverage expansion (Regime-4A) --------
        if self.enable_spawn:
            spawned = self.spawner.step(psi, y_true)
            if spawned:
                self.history["spawns"] += 1

        # -------- 4. Pruning (Regime-4B) --------
        if self.enable_prune:
            pruned = self.pruner.step()
            if pruned:
                self.history["prunes"] += len(pruned)

        # -------- 5. Bookkeeping --------
        self.history["memory_size"].append(
            len(self.memory_bank.class_states)
        )

        return y_hat

    # =================================================
    # Training loop
    # =================================================
    def fit(self, X, y):
        correct = 0
        for psi, label in zip(X, y):
            y_hat = self.step(psi, label)
            if y_hat == label:
                correct += 1
        return correct / len(X)

    # =================================================
    # Prediction
    # =================================================
    def predict_one(self, psi):
        _, score = self.memory_bank.winner(psi)
        return 1 if score >= 0 else -1

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    # =================================================
    # Diagnostics
    # =================================================
    def summary(self):
        return {
            "steps": self.step_count,
            "memory_size": len(self.memory_bank.class_states),
            "num_updates": self.history["updates"],
            "num_spawns": self.history["spawns"],
            "num_prunes": self.history["prunes"],
            "enable_spawn": self.enable_spawn,
            "enable_prune": self.enable_prune,
        }
