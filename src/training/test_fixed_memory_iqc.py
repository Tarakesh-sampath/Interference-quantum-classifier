import numpy as np
import os
from sklearn.metrics import accuracy_score

from src.utils.load_data import load_data
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.backends.hardware_native import HardwareNativeBackend
from src.utils.paths import load_paths
from src.utils.seed import set_seed

def main():
    # -------------------------------------------------
    # Load paths
    # -------------------------------------------------
    set_seed()
    BASE_ROOT, PATHS = load_paths()

    OUT_DIR = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc")
    os.makedirs(OUT_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(OUT_DIR, "fixed_memory_iqc.pkl")
        
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")
    
    # Limit samples for faster testing
    X_train, y_train = X_train[:100], y_train[:100]
    X_test, y_test = X_test[:100], y_test[:100]

    # -------------------------------------------------
    # Quantum-safe normalization (defensive)
    # -------------------------------------------------
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Train Fixed-Memory IQC
    # -------------------------------------------------
    K = 1
    model = FixedMemoryIQC(K=K, eta=0.1, backend=HardwareNativeBackend())#, alpha=0.3, beta=1.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ FixedMemoryIQC | K={K} | Test Accuracy: {acc:.4f}")

    # -------------------------------------------------
    # Save/Load Demonstration
    # -------------------------------------------------
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    print(f"📥 Loading model from {MODEL_PATH}...")
    loaded_model = FixedMemoryIQC.load(MODEL_PATH)

    y_pred_loaded = loaded_model.predict(X_test)
    acc_loaded = accuracy_score(y_test, y_pred_loaded)
    print(f"✅ Loaded Model | Test Accuracy: {acc_loaded:.4f}")

    if np.all(y_pred == y_pred_loaded):
        print("🚀 Success: Predictions match exactly!")
    else:
        print("❌ Error: Predictions do not match!")


if __name__ == "__main__":
    main()
