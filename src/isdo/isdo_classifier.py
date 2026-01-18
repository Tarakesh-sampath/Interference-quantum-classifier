import os
import numpy as np

class ISDOClassifier:
    def __init__(self, proto_dir, K):
        self.proto_dir = proto_dir
        self.K = K
        self.prototypes = {
            0: [np.load(os.path.join(proto_dir, f"class0_proto{i}.npy")) for i in range(K)],
            1: [np.load(os.path.join(proto_dir, f"class1_proto{i}.npy")) for i in range(K)],
        }

    def predict_one(self, psi):
        A0 = sum(np.vdot(p, psi) for p in self.prototypes[0])
        A1 = sum(np.vdot(p, psi) for p in self.prototypes[1])
        return 1 if np.real(A0 - A1) < 0 else 0

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
