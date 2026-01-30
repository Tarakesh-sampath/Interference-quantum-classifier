import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.load_data import load_data
from src.utils.label_utils import ensure_polar
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.inference.weighted_vote_classifier import WeightedVoteClassifier


def main():
    print("\n🚀 Testing Regime-4A (Coverage-Based Adaptive Memory)\n")

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # -------------------------------------------------
    # Initialize memory bank (bootstrap like Regime-3A)
    # -------------------------------------------------
    backend = ExactBackend()

    # Simple bootstrap: one memory per class
    class_states = []

    for cls in [-1, +1]:
        idx = np.where(y_train == cls)[0][0]
        chi0 = X_train[idx].copy()
        chi0 /= np.linalg.norm(chi0)
        class_states.append(ClassState(chi0, label=cls, backend=backend))

    memory_bank = MemoryBank(class_states)

    print("Initial memory size:", len(memory_bank.class_states))

    # -------------------------------------------------
    # Train Regime-4A
    # -------------------------------------------------
    model = Regime4ASpawn(
        memory_bank=memory_bank,
        eta=0.1,
        backend=backend,
        delta_cover=0.2,
        spawn_cooldown=100,
        min_polarized_per_class=1,
    )

    model.fit(X_train, y_train)

    print("\n=== Regime-4A Training Summary ===")
    print(model.summary())

    # -------------------------------------------------
    # Inference (Regime-3B style)
    # -------------------------------------------------
    classifier = WeightedVoteClassifier(memory_bank)

    y_pred = [classifier.predict(x) for x in X_test]
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Regime-4A Evaluation ===")
    print(f"Test Accuracy     : {acc:.4f}")
    print(f"Final Memory Size : {len(memory_bank.class_states)}")

    # -------------------------------------------------
    # Sanity checks
    # -------------------------------------------------
    print("\n=== Sanity Checks ===")

    actions = model.history["action"]
    num_spawned = actions.count("spawned")
    num_updated = actions.count("updated")

    print(f"Spawn events  : {num_spawned}")
    print(f"Update events : {num_updated}")

    if num_spawned == 0:
        print("⚠️  No memories spawned — try lowering delta_cover")
    elif len(memory_bank.class_states) > 50:
        print("⚠️  Memory may be growing too fast")
    else:
        print("✅ Memory growth appears controlled")

    print("\n✅ Regime-4A test completed successfully.\n")


if __name__ == "__main__":
    main()
