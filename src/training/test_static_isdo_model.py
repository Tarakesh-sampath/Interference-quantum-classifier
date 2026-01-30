import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.load_data import load_data
from src.IQL.models.static_isdo_model import StaticISDOModel

def main():
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")

    # -------------------------------------------------
    # Sanity: ensure quantum-safe normalization
    # -------------------------------------------------
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Run Static ISDO Model
    # -------------------------------------------------
    K = 4 # best K from sweep
    model = StaticISDOModel(K=K)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ StaticISDOModel | K={K} | Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
