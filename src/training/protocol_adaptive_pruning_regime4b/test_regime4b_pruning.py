import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime3b_responsible import Regime3BResponsible
from src.IQL.regimes.regime4b_pruning import Regime4BPruning


def main():
    print("\n🚀 Testing Regime-4B (EMA-Based Pruning)\n")

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

    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Initialize memory with extra capacity
    # -------------------------------------------------
    backend = ExactBackend()
    class_states = []

    for cls in [-1, +1]:
        idxs = np.where(y_train == cls)[0][:4]  # 4 per class
        for idx in idxs:
            chi = X_train[idx].astype(np.complex128)
            chi /= np.linalg.norm(chi)
            class_states.append(
                ClassState(chi, backend=backend, label=cls)
            )

    memory_bank = MemoryBank(class_states)

    print("Initial memory size:", len(memory_bank.class_states))

    # -------------------------------------------------
    # Regime-3B (learning)
    # -------------------------------------------------
    learner = Regime3BResponsible(
        memory_bank=memory_bank,
        eta=0.1,
        alpha_correct=0.0,
        alpha_wrong=1.0,
        tau=0.1,
    )

    # -------------------------------------------------
    # Regime-4B (pruning)
    # -------------------------------------------------
    pruner = Regime4BPruning(
        memory_bank=memory_bank,
        tau_harm=-0.15,
        min_age=200,
        min_per_class=1,
        prune_interval=200,
    )

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for step, (psi, label) in enumerate(zip(X_train, y_train)):
        learner.step(psi, label)

        # update metadata
        memory_bank.increment_age()
        memory_bank.update_harm_ema(
            psi,
            y_true=label,
            tau_responsible=0.1,
            beta=0.98,
        )

        pruned = pruner.step()

        if pruned:
            print(
                f"Step {step}: pruned {len(pruned)} memories "
                f"(current size = {len(memory_bank.class_states)})"
            )

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    y_pred = [learner.predict_one(x) for x in X_test]
    test_acc = accuracy_score(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"Test Accuracy     : {test_acc:.4f}")
    print(f"Final Memory Size : {len(memory_bank.class_states)}")

    print("\n=== Pruning Summary ===")
    print(pruner.summary())

    print("\n✅ Regime-4B pruning test completed.\n")


if __name__ == "__main__":
    main()
