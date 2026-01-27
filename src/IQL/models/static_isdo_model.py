# src/IQL/models/static_isdo_model.py

from src.IQL.baselines.static_isdo_classifier import StaticISDOClassifier
from src.utils.paths import load_paths
from src.IQL.learning.calculate_prototype import generate_prototypes
import os

class StaticISDOModel:
    """
    Static ISDO Model (Baseline)

    - K prototypes per class
    - No learning
    - Fixed interference reference state |chi>
    """

    def __init__(self, K: int):
        _, PATHS = load_paths()
        self.proto_dir = PATHS["class_prototypes"]
        self.K = K
        self.classifier = None

    def _ensure_prototypes(self, X, y):
        """
        Generate prototypes if they do not already exist.
        """
        _, PATHS = load_paths()
        proto_base = PATHS["class_prototypes"]
        proto_dir = os.path.join(proto_base, f"K{self.K}")
        os.makedirs(proto_dir, exist_ok=True)
        generate_prototypes(
            X=X,
            y=y,
            K=self.K,
            output_dir=proto_dir,
            seed = 42
        )
    
    def fit(self,X,y):
        """
        Offline preparation only.
        Loads precomputed prototypes and builds classifier.
        """
        self._ensure_prototypes(X,y)
        self.classifier = StaticISDOClassifier(
            proto_dir=self.proto_dir,
            K=self.K
        )
        return self

    def predict(self, X):
        if self.classifier is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.classifier.predict(X)
