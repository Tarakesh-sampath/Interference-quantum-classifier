from ..learning.regime2_update import regime2_update

class Regime2Trainer:
    def __init__(self, chi_init, eta):
        self.chi = chi_init
        self.eta = eta
        self.num_updates = 0

    def step(self, psi, y):
        s = self.chi.score(psi)
        y_hat = 1 if s >= 0 else -1

        if y_hat != y:
            self.chi.vector, updated = regime2_update(
                self.chi.vector, psi, y, self.eta
            )
            if updated:
                self.num_updates += 1

        return y_hat, s

    def train(self, dataset):
        correct = 0
        for psi, y in dataset:
            y_hat, _ = self.step(psi, y)
            if y_hat == y:
                correct += 1
        return correct / len(dataset)
