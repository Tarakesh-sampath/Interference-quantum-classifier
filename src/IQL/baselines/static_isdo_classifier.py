import os
import numpy as np
from tqdm import tqdm
from src.IQL.backends.exact import ExactBackend
from src.IQL.learning.prototype import load_prototypes

class StaticISDOClassifier:
    def __init__(self, proto_dir, K):
        self.proto_dir = proto_dir
        self.K = K
        self.exact = ExactBackend()
        protos = load_prototypes(
            K=K,
            output_dir=os.path.join(proto_dir, f"K{K}")
        )
        # Binary split (ignore labels even if present)
        self.prototypes = {0: [], 1: []}
        for p in protos:
            # class index is encoded in filename order,
            # OR we can rely on p["label"] if present
            cls = p["label"] if p["label"] is not None else None
            self.prototypes[cls].append(p["vector"])
        self.chi = sum(self.prototypes[0]) - sum(self.prototypes[1])
        self.chi /= np.linalg.norm(self.chi)
    def predict_one(self, psi):
        return 1 if self.exact.score(self.chi, psi) < 0 else 0

    def predict(self, X):
        return np.array([self.predict_one(x) for x in tqdm(X, desc="ISDO Prediction", leave=False)])
