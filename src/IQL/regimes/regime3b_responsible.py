import numpy as np
from src.IQL.learning.update import update
from src.IQL.backends.exact import ExactBackend


class Regime3BResponsible:
    """
    Regime-3B: Responsible-Set Corrective Learning

    - Same as Regime-3A, but:
      instead of updating only the winner,
      update all RESPONSIBLE memories.

    - Direction still comes from y_true
    - Uses existing update() primitive
    - Guard-A: update energy normalized by |responsible set|
    """

    def __init__(
        self,
        memory_bank,
        eta,
        backend=ExactBackend(),
        alpha_correct: float = 0.0,
        alpha_wrong: float = 1.0,
        tau: float = 0.1,   # responsibility threshold
    ):
        self.memory_bank = memory_bank
        self.eta = eta
        self.backend = backend

        self.alpha_correct = alpha_correct
        self.alpha_wrong = alpha_wrong
        self.tau = tau

        self.num_updates = 0

    # -------------------------------------------------
    # Core step
    # -------------------------------------------------
    def step(self, psi, y_true):
        # Compute all scores
        scores = self.memory_bank.scores(psi)

        # Winner (for prediction only)
        idx_star = int(np.argmax(np.abs(scores)))
        score_star = scores[idx_star]

        # Correctness
        misclassified = (y_true * score_star) < 0
        alpha = self.alpha_wrong if misclassified else self.alpha_correct

        # Prediction
        y_hat = 1 if score_star >= 0 else -1

        # ----------------------------------
        # Responsible set
        # ----------------------------------
        responsible = [
            cs for cs, s in zip(self.memory_bank.class_states, scores)
            if abs(s) > self.tau
        ]

        # Fallback: always update winner at least
        if not responsible:
            responsible = [self.memory_bank.class_states[idx_star]]

        # Guard-A normalization
        scale = self.eta * alpha / len(responsible)

        # ----------------------------------
        # Update ALL responsible memories
        # ----------------------------------
        for cs in responsible:
            chi_new, updated = update(
                cs.vector,
                psi,
                y_true,     # ← direction = truth (unchanged)
                scale,
                self.backend,
            )
            if updated:
                cs.vector = chi_new
                self.num_updates += 1

        return y_hat

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    def fit(self, X, y):
        correct = 0
        for psi, label in zip(X, y):
            y_hat = self.step(psi, label)
            if y_hat == label:
                correct += 1
        return correct / len(X)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict_one(self, psi):
        _, score = self.memory_bank.winner(psi)
        return 1 if score >= 0 else -1

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    def summary(self):
        return {
            "memory_size": len(self.memory_bank.class_states),
            "num_updates": self.num_updates,
            "tau": self.tau,
            "alpha_correct": self.alpha_correct,
            "alpha_wrong": self.alpha_wrong,
        }
