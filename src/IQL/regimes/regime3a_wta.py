from src.IQL.learning.update import update
from src.IQL.backends.exact import ExactBackend
import pickle
from tqdm import tqdm


class WinnerTakeAll:
    """
    Regime 3-A: Winner-Takes-All IQC (α-scaled formulation)

    Special case:
        alpha_correct = 0
        alpha_wrong   = 1
    reproduces the original Regime-3A exactly.
    """

    def __init__(
        self,
        memory_bank,
        eta,
        backend=ExactBackend(),
        alpha: float = 0.0,
        beta: float = 1.0,
    ):
        self.memory_bank = memory_bank
        self.eta = eta
        self.backend = backend

        # Scaling factors
        self.alpha_correct = alpha
        self.alpha_wrong = beta

        self.num_updates = 0

        self.history = {
            "winner_idx": [],
            "scores": [],
            "updates": [],
            "alpha": [],
        }

    def step(self, psi, y):
        # ----------------------------------
        # Winner selection (unchanged)
        # ----------------------------------
        idx, score = self.memory_bank.winner(psi)
        cs = self.memory_bank.class_states[idx]

        # ----------------------------------
        # Correctness check
        # ----------------------------------
        misclassified = (y * score) < 0

        # α-scaling (THIS is the only real change)
        alpha = self.alpha_wrong if misclassified else self.alpha_correct

        # ----------------------------------
        # Scaled update (winner only)
        # ----------------------------------
        chi_new, updated = update(
            cs.vector,
            psi,
            y,
            self.eta * alpha,
            self.backend,
        )

        if updated:
            cs.vector = chi_new
            self.num_updates += 1

        # Prediction (unchanged)
        y_hat = 1 if score >= 0 else -1

        # ----------------------------------
        # Logging
        # ----------------------------------
        self.history["winner_idx"].append(idx)
        self.history["scores"].append(score)
        self.history["updates"].append(updated)
        self.history["alpha"].append(alpha)

        return y_hat, idx, updated

    def fit(self, X, y):
        correct = 0
        pbar = tqdm(zip(X, y), total=len(X), desc="Training")
        for x, label in pbar:
            y_hat, _, _ = self.step(x, label)
            if y_hat == label:
                correct += 1
            pbar.set_postfix({"acc": f"{correct/(pbar.n+1):.4f}"})
        return correct / len(X)

    def predict_one(self, X):
        _, score = self.memory_bank.winner(X)
        return 1 if score >= 0 else -1

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def save(self, path):
        """
        Save trained memory bank and history.
        """
        payload = {
            "memory_bank": self.memory_bank,
            "eta": self.eta,
            "num_updates": self.num_updates,
            "history": self.history,
            "backend": self.backend,
            "alpha_correct": self.alpha_correct,
            "alpha_wrong": self.alpha_wrong,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        """
        Load a trained Winner-Take-All model.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            memory_bank=payload["memory_bank"],
            eta=payload["eta"],
            backend=payload["backend"],
            alpha_correct=payload.get("alpha_correct", 0.0),
            alpha_wrong=payload.get("alpha_wrong", 1.0),
        )

        obj.num_updates = payload["num_updates"]
        obj.history = payload["history"]

        return obj
