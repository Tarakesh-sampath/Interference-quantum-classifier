# src/IQL/training/adaptive_memory_trainer.py
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning

class AdaptiveMemoryModel:
    """
    System-level adaptive controller for IQC.

    Responsibilities:
    - Enforce memory lifecycle invariants
    - Orchestrate learning (Regime-4A)
    - Handle aging, harm tracking, and pruning (Regime-4B)

    This class contains NO learning logic.
    """

    def __init__(
        self,
        memory_bank,
        learner : Regime4ASpawn,          # Regime4ASpawn
        pruner : Regime4BPruning,           # Regime4BPruning
        tau_responsible: float = 0.1,
        beta: float = 0.98,
    ):
        self.memory_bank = memory_bank
        self.learner = learner
        self.pruner = pruner
        self.tau_responsible = tau_responsible
        self.beta = beta

        self.step_count = 0
        self.history = {
            "action": [],        # spawned | updated | noop
            "memory_size": [],
            "num_pruned": [],
        }

    def step(self, psi, y):
        """
        Execute ONE adaptive training step.

        Ordering is STRICT and MUST NOT be changed.
        """

        # -------------------------------------------------
        # STEP 1: learning + possible memory spawn
        # -------------------------------------------------
        action = self.learner.step(psi, y)

        # -------------------------------------------------
        # STEP 2: age update (MANDATORY)
        # -------------------------------------------------
        self.memory_bank.increment_age()

        # -------------------------------------------------
        # STEP 3: harm EMA update (MANDATORY)
        # -------------------------------------------------
        self.memory_bank.update_harm_ema(
            psi,
            tau_responsible=self.tau_responsible,
            beta=self.beta,
        )

        # -------------------------------------------------
        # STEP 4: pruning (PERIODIC)
        # -------------------------------------------------
        pruned = self.pruner.step()
        num_pruned = len(pruned) if pruned else 0

        # -------------------------------------------------
        # STEP 5: bookkeeping
        # -------------------------------------------------
        self.step_count += 1
        self.history["action"].append(action)
        self.history["memory_size"].append(
            len(self.memory_bank.class_states)
        )
        self.history["num_pruned"].append(num_pruned)

        return action, pruned

    def fit(self, X, y):
        """
        Online adaptive training loop.
        """
        for psi, label in zip(X, y):
            self.step(psi, label)
        return self

    def predict(self, X):
        """
        Inference using winner-take-all interference.
        """
        preds = []
        for psi in X:
            _, score = self.memory_bank.winner(psi)
            preds.append(1 if score >= 0 else -1)
        return preds

    def summary(self):
        return {
            "steps": self.step_count,
            "final_memory_size": len(self.memory_bank.class_states),
            "num_spawns": self.history["action"].count("spawned"),
            "num_pruned": sum(self.history["num_pruned"]),
        }
