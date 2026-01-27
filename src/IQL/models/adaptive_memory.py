import numpy as np
from collections import deque
from src.IQL.learning.update import update
from src.IQL.backends.exact import ExactBackend
import pickle

class AdaptiveMemory:
    """
    Regime 3-C: Dynamic Memory Growth with Percentile-based τ
    """

    def __init__(
        self,
        memory_bank,
        eta=0.1,
        percentile=5,
        tau_abs = -0.4,
        margin_window=500,
        backend=ExactBackend()
    ):
        self.memory_bank = memory_bank
        self.eta = eta
        self.percentile = percentile
        self.tau_abs = tau_abs
        self.backend = backend

        # store recent margins
        self.margins = deque(maxlen=margin_window)

        self.num_updates = 0
        self.num_spawns = 0

        self.history = {
            "margin": [],
            "spawned": [],
            "num_memories": [],
        }

    def aggregated_score(self, psi):
        scores = self.memory_bank.scores(psi)
        return sum(scores) / len(scores)

    def step(self, psi, y):
        S = self.aggregated_score(psi)
        margin = y * S

        # collect negative margins only
        neg_margins = [m for m in self.margins if m < 0]

        spawned = False

        # compute percentile only if we have enough negative history
        if len(neg_margins) >= 20:
            tau = np.percentile(neg_margins, self.percentile)

            if margin < tau:
                # 🔥 spawn new memory
                chi_new = y * psi
                chi_new = chi_new / np.linalg.norm(chi_new)
                self.memory_bank.add_memory(chi_new)
                self.num_spawns += 1
                spawned = True

        # otherwise, normal Regime-2 update on winner
        if not spawned and margin < 0:
            idx, _ = self.memory_bank.winner(psi)
            cs = self.memory_bank.class_states[idx]

            chi_new, updated = update(
                cs.vector, psi, y, self.eta, self.backend
            )

            if updated:
                cs.vector = chi_new
                self.num_updates += 1

        # logging
        self.margins.append(margin)
        self.history["margin"].append(margin)
        self.history["spawned"].append(spawned)
        self.history["num_memories"].append(len(self.memory_bank.class_states))

        return margin, spawned
    
    def memory_size(self):
        return len(self.memory_bank.class_states)

    def fit(self, X, y):
        for psi, y in zip(X, y):
            self.step(psi, y)

    def predict_one(self, X):
        _, score = self.memory_bank.winner(X)
        return 1 if score >= 0 else -1
    
    def predict(self, X):
        return [self.predict_one(x) for x in X]
        
    def save(self, path):
        """
        Save trained memory + training history.
        """
        payload = {
            "memory_bank": self.memory_bank,
            "eta": self.eta,
            "percentile": self.percentile,
            "tau_abs": self.tau_abs,
            "margins": list(self.margins),
            "num_updates": self.num_updates,
            "num_spawns": self.num_spawns,
            "history": self.history,
            "backend": self.backend,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        """
        Load a previously trained Regime-3C model.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            memory_bank=payload["memory_bank"],
            eta=payload["eta"],
            percentile=payload["percentile"],
            tau_abs=payload["tau_abs"],
            margin_window=len(payload["margins"]),
            backend=payload["backend"],
        )

        # restore training state
        from collections import deque
        obj.margins = deque(payload["margins"], maxlen=len(payload["margins"]))
        obj.num_updates = payload["num_updates"]
        obj.num_spawns = payload["num_spawns"]
        obj.history = payload["history"]

        return obj