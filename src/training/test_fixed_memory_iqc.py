import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.load_data import load_data
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC


def main():
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")

    # -------------------------------------------------
    # Quantum-safe normalization (defensive)
    # -------------------------------------------------
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Train Fixed-Memory IQC
    # -------------------------------------------------
    K = 1
    model = FixedMemoryIQC(K=K, eta=0.1)#, alpha=0.3, beta=1.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ FixedMemoryIQC | K={K} | Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
