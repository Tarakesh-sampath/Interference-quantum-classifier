# src/IQL/models/fixed_memory_iqc.py

import os
import numpy as np

from src.utils.paths import load_paths
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier
from src.IQL.backends.exact import ExactBackend
from src.IQL.learning.calculate_prototype import generate_prototypes
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

    def __init__(self, K: int, eta: float = 0.1, backend=None):
        self.K = K
        self.eta = eta
        self.backend = backend or ExactBackend()

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

    def _load_prototypes(self):
        _, PATHS = load_paths()
        proto_dir = PATHS["class_prototypes"]

        vectors = []
        for cls in [0, 1]:
            for i in range(self.K):
                path = os.path.join(
                    proto_dir, f"K{self.K}", f"class{cls}_proto{i}.npy"
                )
                vectors.append(np.load(path))
        return vectors

    def fit(self, X, y):
        # -------------------------------------------------
        # Step 1: ensure prototypes exist
        # -------------------------------------------------
        self._ensure_prototypes(X, y)

        # -------------------------------------------------
        # Step 2: initialize memory bank
        # -------------------------------------------------
        proto_vectors = self._load_prototypes()
        class_states = [
            ClassState(v, backend=self.backend)
            for v in proto_vectors
        ]
        self.memory_bank = MemoryBank(class_states)

        # -------------------------------------------------
        # Step 3: Regime-3A training
        # -------------------------------------------------
        self.trainer = WinnerTakeAll(
            memory_bank=self.memory_bank,
            eta=self.eta,
            backend=self.backend
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
