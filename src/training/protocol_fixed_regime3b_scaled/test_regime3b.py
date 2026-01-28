import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime3b_scaled import Regime3BScaled
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier


def main():
    print("\n🚀 Testing Regime-3B (Winner-only, Scaled Update)\n")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Initialize memory (same as Regime-3A baseline)
    # --------------------------------------------------
    backend = ExactBackend()

    class_states = []
    for cls in [-1, +1]:
        idx = np.where(y_train == cls)[0][0]
        chi0 = X_train[idx].astype(np.complex128)
        chi0 /= np.linalg.norm(chi0)
        class_states.append(ClassState(chi0, backend=backend))

    memory_bank = MemoryBank(class_states)

    print("Initial memory size:", len(memory_bank.class_states))

    # --------------------------------------------------
    # Train Regime-3B
    # --------------------------------------------------
    model = Regime3BScaled(
        memory_bank=memory_bank,
        eta=0.1,
        backend=backend,
        alpha_correct=0.3,
        alpha_wrong=1.5,
    )

    model.fit(X_train, y_train)

    print("\n=== Regime-3B Training Summary ===")
    print(model.summary())

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    clf = WeightedVoteClassifier(memory_bank)
    y_pred = [clf.predict(x) for x in X_test]

    acc = accuracy_score(y_test, y_pred)

    print("\n=== Regime-3B Evaluation ===")
    print(f"Test Accuracy     : {acc:.4f}")
    print(f"Final Memory Size : {len(memory_bank.class_states)}")

    print("\n✅ Regime-3B test completed.\n")


if __name__ == "__main__":
    main()
