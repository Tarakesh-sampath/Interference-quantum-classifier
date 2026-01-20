import numpy as np

class Regime3BClassifier:
    """
    Regime 3-B: Interference Voting Classifier
    Uses multiple learned |chi> states with soft aggregation.
    """

    def __init__(self, memory_bank, weights=None):
        self.memory_bank = memory_bank
        self.M = len(memory_bank.class_states)

        if weights is None:
            self.weights = np.ones(self.M) / self.M
        else:
            self.weights = np.asarray(weights)
            assert len(self.weights) == self.M
            self.weights = self.weights / np.sum(self.weights)

    def score(self, psi):
        scores = np.array([
            float(np.real(np.vdot(cs.vector, psi)))
            for cs in self.memory_bank.class_states
        ])
        return float(np.dot(self.weights, scores))

    def predict(self, psi):
        s = self.score(psi)
        return 1 if s >= 0 else -1
