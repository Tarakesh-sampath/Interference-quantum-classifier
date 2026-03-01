# src/IQL/models/fixed_memory_iqc.py

import os
import pickle
import numpy as np

from src.utils.paths import load_paths
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier
from src.IQL.backends.exact import ExactBackend
from src.IQL.learning.prototype import generate_prototypes,load_prototypes
from src.utils.label_utils import ensure_binary


class FixedMemoryIQC:
    """
    Fixed-Memory Interference Quantum Classifier (IQC)

    Training pipeline:
    1. Generate K prototypes per class (if missing)
    2. Initialize K×2 quantum memory states
    3. Train with Winner-Take-All (Regime-3A)
    4. Freeze memory
    """

    def __init__(self, K: int, eta: float = 0.1, backend=None, alpha: float = 0, beta: float = 1):
        self.K = K
        self.eta = eta
        self.backend = backend or ExactBackend()
        self.alpha = alpha
        self.beta = beta

        self.memory_bank = None
        self.trainer = None
        self.classifier = None

    def _ensure_prototypes(self, X, y):
        """
        Generate prototypes if they do not already exist.
        """
        _, PATHS = load_paths()
        proto_base = PATHS["class_prototypes"]
        proto_dir = os.path.join(proto_base, f"K{self.K}")

        os.makedirs(proto_dir, exist_ok=True)
        y_binary = ensure_binary(y)
        generate_prototypes(
            X=X,
            y=y_binary,
            K=self.K,
            output_dir=proto_dir
        )
        return load_prototypes(K=self.K, output_dir=proto_dir)
        
    def fit(self, X, y):
        # -------------------------------------------------
        # Step 1: ensure prototypes exist
        # -------------------------------------------------
        proto = self._ensure_prototypes(X, y)

        # -------------------------------------------------
        # Step 2: initialize memory bank
        # -------------------------------------------------
        class_states = [
            ClassState(v["vector"], backend=self.backend, label=v["label"])
            for v in proto
        ]
        self.memory_bank = MemoryBank(class_states)

        # -------------------------------------------------
        # Step 3: Regime-3A training
        # -------------------------------------------------
        self.trainer = WinnerTakeAll(
            memory_bank=self.memory_bank,
            eta=self.eta,
            backend=self.backend,
            alpha = self.alpha,
            beta = self.beta
        )
        self.trainer.fit(X, y)

        # -------------------------------------------------
        # Step 4: freeze → inference
        # -------------------------------------------------
        self.classifier = WeightedVoteClassifier(self.memory_bank)
        return self

    def predict(self, X):
        if self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return [self.classifier.predict(x) for x in X]

    def save(self, path):
        """
        Save the model state.
        """
        payload = {
            "K": self.K,
            "eta": self.eta,
            "alpha": self.alpha,
            "beta": self.beta,
            "memory_bank": self.memory_bank,
            "classifier": self.classifier,
            "backend": self.backend
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        """
        Load the model state.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            K=payload["K"],
            eta=payload["eta"],
            alpha=payload["alpha"],
            beta=payload["beta"],
            backend=payload.get("backend")
        )
        obj.memory_bank = payload["memory_bank"]
        obj.classifier = payload["classifier"]
        return obj
