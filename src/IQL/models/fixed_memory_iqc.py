# src/IQL/models/fixed_memory_iqc.py

import os
import numpy as np

from src.utils.paths import load_paths
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier
from src.IQL.backends.exact import ExactBackend


class FixedMemoryIQC:
    """
    Fixed-Memory Interference Quantum Classifier (IQC)

    - Initializes K quantum memory states from class prototypes
    - Trains using Winner-Take-All (Regime-3A)
    - Memory count is fixed (no growth)
    - Inference via weighted interference voting
    """

    def __init__(self, K: int, eta: float = 0.1, backend=None):
        self.K = K
        self.eta = eta
        self.backend = backend or ExactBackend()

        self.memory_bank = None
        self.trainer = None
        self.classifier = None

    def _load_prototypes(self):
        """
        Load K prototypes per class and return list of vectors.
        """
        _, PATHS = load_paths()
        proto_dir = PATHS["class_prototypes"]

        prototypes = []
        for cls in [0, 1]:
            for i in range(self.K):
                path = os.path.join(
                    proto_dir, f"K{self.K}", f"class{cls}_proto{i}.npy"
                )
                prototypes.append(np.load(path))
        return prototypes

    def fit(self, X, y):
        """
        Train fixed-memory IQC.
        """
        # -------------------------------------------------
        # Initialize memory from prototypes
        # -------------------------------------------------
        proto_vectors = self._load_prototypes()

        class_states = [
            ClassState(vec, backend=self.backend)
            for vec in proto_vectors
        ]

        self.memory_bank = MemoryBank(class_states)

        # -------------------------------------------------
        # Regime-3A: Winner-Take-All learning
        # -------------------------------------------------
        self.trainer = WinnerTakeAll(
            memory_bank=self.memory_bank,
            eta=self.eta,
            backend=self.backend,
        )
        self.trainer.fit(X, y)

        # -------------------------------------------------
        # Freeze memory → inference-only
        # -------------------------------------------------
        self.classifier = WeightedVoteClassifier(self.memory_bank)
        return self

    def predict(self, X):
        if self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return [self.classifier.predict(x) for x in X]
