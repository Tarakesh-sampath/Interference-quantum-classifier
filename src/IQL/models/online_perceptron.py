import numpy as np
from src.IQL.learning.update import update
from src.IQL.backends.base import InterferenceBackend
import pickle

class OnlinePerceptron:
    """
    Online Interference Quantum Classifier (Regime 2)

    Fixed circuit.
    Trainable object: |chi>
    """

    def __init__(self, class_state, eta: float, backend: InterferenceBackend):
        self.class_state = class_state
        self.eta = eta
        self.backend = backend
        # logs
        self.num_updates = 0
        self.history = {
            "scores": [],
            "margins": [],
            "updates": [],
        }

    def step(self, psi: np.ndarray, y: int):
        """
        Process a single training example.
        """
        s = self.class_state.score(psi)
        margin = y * s
        y_hat = 1 if s >= 0 else -1

        chi_new, updated = update(
            self.class_state.vector, psi, y, self.eta, self.backend
        )

        if updated:
            self.class_state.vector = chi_new
            self.num_updates += 1

        # logging
        self.history["scores"].append(s)
        self.history["margins"].append(margin)
        self.history["updates"].append(updated)

        return y_hat, s, updated

    def fit(self, X, y):
        """
        Single-pass online training.
        dataset: iterable of (psi, y)
        """
        correct = 0

        for i in range(len(X)):
            y_hat, _, _ = self.step(X[i], y[i])
            if y_hat == y[i]:
                correct += 1

        accuracy = correct / len(X)
        return accuracy
    
    def predict_one(self, X):
        s = self.class_state.score(X)
        return 1 if s >= 0 else -1
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def save(self, path):
        """
        Save trained perceptron state and history.
        """
        payload = {
            "class_state": self.class_state,
            "eta": self.eta,
            "num_updates": self.num_updates,
            "history": self.history,
            "backend": self.backend,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        """
        Load a trained perceptron model.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            class_state=payload["class_state"],
            eta=payload["eta"],
            backend=payload["backend"],
        )

        # restore training statistics
        obj.num_updates = payload["num_updates"]
        obj.num_mistakes = payload["num_mistakes"]
        obj.margin_history = payload["margin_history"]
        obj.history = payload["history"]

        return obj