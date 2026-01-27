import numpy as np

from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.regimes.regime3c_adaptive import AdaptiveMemory
from src.IQL.regimes.regime3a_wta import WinnerTakeAll
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier
from src.IQL.backends.exact import ExactBackend
from src.IQL.learning.calculate_prototype import generate_prototypes
from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar, ensure_binary

import os


class AdaptiveIQC:
    """
    Final Adaptive Interference Quantum Classifier (IQC)

    Pipeline:
    1. Prototype generation (offline, classical)
    2. Adaptive growth (Regime-3C)
    3. Consolidation (Regime-3A)
    4. Inference-only classifier
    """

    def __init__(
        self,
        K_init=3,
        eta=0.1,
        percentile=5,
        backend=None,
        consolidate=True,
    ):
        self.K_init = K_init
        self.eta = eta
        self.percentile = percentile
        self.backend = backend or ExactBackend()
        self.consolidate = consolidate

        self.memory_bank = None
        self.regime3c = None
        self.classifier = None

    def _initialize_memory(self, X, y):
        _, PATHS = load_paths()
        proto_dir = os.path.join(PATHS["class_prototypes"], f"K{self.K_init}")

        y_binary = ensure_binary(y)
        generate_prototypes(
            X=X,
            y=y_binary,
            K=self.K_init,
            output_dir=proto_dir,
            seed =42,
        )

        class_states = []
        for cls in [0, 1]:
            for i in range(self.K_init):
                vec = np.load(
                    os.path.join(proto_dir, f"class{cls}_proto{i}.npy")
                )
                class_states.append(
                    ClassState(vec, backend=self.backend)
                )

        self.memory_bank = MemoryBank(class_states)

    def fit(self, X, y):
        # -------------------------------------------------
        # Step 1 — initialize memory
        # -------------------------------------------------
        y = ensure_polar(y)
        self._initialize_memory(X, y)

        # -------------------------------------------------
        # Step 2 — adaptive growth + pruning (Regime-3C)
        # -------------------------------------------------
        self.regime3c = AdaptiveMemory(
            memory_bank=self.memory_bank,
            eta=self.eta,
            percentile=self.percentile,
            backend=self.backend,
        )
        self.regime3c.fit(X, y)

        # -------------------------------------------------
        # Step 3 — optional consolidation (Regime-3A)
        # -------------------------------------------------
        if self.consolidate:
            consolidator = WinnerTakeAll(
                memory_bank=self.memory_bank,
                eta=self.eta,
                backend=self.backend,
            )
            consolidator.fit(X, y)

        # -------------------------------------------------
        # Step 4 — freeze & inference
        # -------------------------------------------------
        self.classifier = WeightedVoteClassifier(self.memory_bank)
        return self

    def predict(self, X):
        if self.classifier is None:
            raise RuntimeError("Model not trained.")
        return [self.classifier.predict(x) for x in X]
