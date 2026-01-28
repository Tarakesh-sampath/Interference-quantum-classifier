import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.models.adaptive_iqc import AdaptiveIQC


def main():
    print("\n🚀 Testing AdaptiveIQC (Regime-3A + 4A + 4B)\n")

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    _, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]

    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y = np.load(os.path.join(EMBED_DIR, "val_labels_polar.npy"))

    train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    y_train = ensure_polar(y_train)
    y_test = ensure_polar(y_test)

    # Normalize embeddings → quantum states
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Initialize memory (minimal polarized start)
    # -------------------------------------------------
    backend = ExactBackend()
    class_states = []

    for cls in [-1, +1]:
        idx = np.where(y_train == cls)[0][0]
        chi = X_train[idx].astype(np.complex128)
        chi /= np.linalg.norm(chi)

        class_states.append(
            ClassState(
                vector=chi,
                backend=backend,
                label=cls,
            )
        )

    memory_bank = MemoryBank(class_states)

    print("Initial memory size:", len(memory_bank.class_states))

    # -------------------------------------------------
    # Build AdaptiveIQC
    # -------------------------------------------------
    model = AdaptiveIQC(
        memory_bank=memory_bank,

        # Regime-3A
        eta=0.1,
        alpha_correct=0.0,
        alpha_wrong=1.0,

        # Regime-4A
        tau_spawn=0.1,
        enable_spawn=True,

        # Regime-4B
        tau_harm=-0.15,
        min_age=200,
        min_per_class=1,
        prune_interval=200,
        tau_responsible=0.1,
        harm_ema_beta=0.98,
        enable_prune=True,
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    train_acc = model.fit(X_train, y_train)

    print("\n=== Training Summary ===")
    print(model.summary())
    print(f"Train Accuracy     : {train_acc:.4f}")
    print(f"Final Memory Size  : {len(memory_bank.class_states)}")

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"Test Accuracy      : {test_acc:.4f}")
    print(f"Memory Size (test) : {len(memory_bank.class_states)}")

    print("\n✅ AdaptiveIQC test completed successfully.\n")


if __name__ == "__main__":
    main()
