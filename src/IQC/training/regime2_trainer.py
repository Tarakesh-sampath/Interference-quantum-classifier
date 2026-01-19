import numpy as np
from ..learning.regime2_update import regime2_update
from ..observable.isdo_score import isdo_score


class Regime2Trainer:
    """
    Online Interference Quantum Classifier (Regime 2)

    Fixed circuit.
    Trainable object: |chi>
    """

    def __init__(self, class_state, eta: float):
        self.class_state = class_state
        self.eta = eta

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
        chi_vec = self.class_state.vector
        s = isdo_score(chi_vec, psi)
        margin = y * s
        y_hat = 1 if s >= 0 else -1

        chi_new, updated = regime2_update(
            chi_vec, psi, y, self.eta
        )

        if updated:
            self.class_state.vector = chi_new
            self.num_updates += 1

        # logging
        self.history["scores"].append(s)
        self.history["margins"].append(margin)
        self.history["updates"].append(updated)

        return y_hat, s, updated

    def train(self, dataset):
        """
        Single-pass online training.
        dataset: iterable of (psi, y)
        """
        correct = 0

        for psi, y in dataset:
            y_hat, _, _ = self.step(psi, y)
            if y_hat == y:
                correct += 1

        accuracy = correct / len(dataset)
        return accuracy
