from ..learning.perceptron_update import perceptron_update

class WinnerTakeAllTrainer:
    """
    Regime 3-A: Winner-Takes-All IQC
    Only the winning memory is updated.
    """

    def __init__(self, memory_bank, eta):
        self.memory_bank = memory_bank
        self.eta = eta
        self.num_updates = 0

        self.history = {
            "winner_idx": [],
            "scores": [],
            "updates": [],
        }

    def step(self, psi, y):
        idx, score = self.memory_bank.winner(psi)
        cs = self.memory_bank.class_states[idx]

        chi_new, updated = perceptron_update(
            cs.vector, psi, y, self.eta
        )

        if updated:
            cs.vector = chi_new
            self.num_updates += 1

        y_hat = 1 if score >= 0 else -1

        # logging
        self.history["winner_idx"].append(idx)
        self.history["scores"].append(score)
        self.history["updates"].append(updated)

        return y_hat, idx, updated

    def fit(self, X, y):
        correct = 0
        for x, y in zip(X, y):
            y_hat, _, _ = self.step(x, y)
            if y_hat == y:
                correct += 1
        return correct / len(X)

    
    def predict_one(self, X):
        _, score = self.memory_bank.winner(X)
        return 1 if score >= 0 else -1
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]