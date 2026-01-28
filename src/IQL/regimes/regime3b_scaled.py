import numpy as np
from collections import defaultdict

from src.IQL.learning.update import update
from src.IQL.backends.exact import ExactBackend


class Regime3BScaled:
    """
    Regime-3B: Winner-only, confidence-scaled interference learning

    - Same winner selection as Regime-3A
    - Winner-only update (SAFE)
    - Update magnitude depends on global correctness
    - Memory labels stored locally (no ClassState change)
    """

    def __init__(
        self,
        memory_bank,
        eta: float = 0.1,
        backend=None,
        alpha_correct: float = 0.3,
        alpha_wrong: float = 1.5,
    ):
        """
        Args:
            memory_bank (MemoryBank)
            eta (float): base learning rate
            backend (InterferenceBackend)
            alpha_correct (float): scale if prediction is correct
            alpha_wrong (float): scale if prediction is wrong
        """
        self.memory_bank = memory_bank
        self.eta = eta
        self.backend = backend or ExactBackend()

        self.alpha_correct = alpha_correct
        self.alpha_wrong = alpha_wrong

        # Local memory labels (index → label)
        self.memory_labels = {}

        # Initialize labels for existing memories
        for i, cs in enumerate(self.memory_bank.class_states):
            # Assume initial memories are class-polarized by construction
            self.memory_labels[i] = None  # filled lazily

        # Stats
        self.num_updates = 0
        self.num_correct = 0
        self.num_wrong = 0

    # --------------------------------------------------
    # Core step
    # --------------------------------------------------
    def step(self, psi: np.ndarray, y: int):
        """
        One learning step.

        Args:
            psi (np.ndarray): input state
            y (int): label in {-1, +1}

        Returns:
            str: "updated" or "noop"
        """
        # Winner (exact Regime-3A semantics)
        winner_idx, s_star = self.memory_bank.winner(psi)
        correct = (y * s_star) > 0

        # Select scaling factor
        alpha = self.alpha_correct if correct else self.alpha_wrong

        if correct:
            self.num_correct += 1
        else:
            self.num_wrong += 1

        # Apply scaled update to winner ONLY
        cs = self.memory_bank.class_states[winner_idx]

        chi_new, updated = update(
            cs.vector,
            psi,
            y,
            eta=self.eta * alpha,
            backend=self.backend,
        )

        if updated:
            cs.vector = chi_new
            self.num_updates += 1

            # Assign label if not yet set
            if self.memory_labels.get(winner_idx) is None:
                self.memory_labels[winner_idx] = y

            return "updated"

        return "noop"

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    def fit(self, X, y):
        for psi, label in zip(X, y):
            self.step(psi, label)
        return self

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    def summary(self):
        return {
            "memory_size": len(self.memory_bank.class_states),
            "num_updates": self.num_updates,
            "num_correct": self.num_correct,
            "num_wrong": self.num_wrong,
            "alpha_correct": self.alpha_correct,
            "alpha_wrong": self.alpha_wrong,
        }


