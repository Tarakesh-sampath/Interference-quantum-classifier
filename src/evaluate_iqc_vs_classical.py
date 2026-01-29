import os
import numpy as np
from sklearn.metrics import accuracy_score

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Utilities
from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar, ensure_binary

# Adaptive IQC
from src.IQL.models.adaptive_memory_model import AdaptiveMemoryModel
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning


def load_data():
    _, PATHS = load_paths()
    EMBED_DIR = PATHS["embeddings"]

    X = np.load(os.path.join(EMBED_DIR, "val_embeddings.npy"))
    y_bin = np.load(os.path.join(EMBED_DIR, "val_labels.npy"))
    y_pol = np.load(os.path.join(EMBED_DIR, "val_labels_polar.npy"))

    train_idx = np.load(os.path.join(EMBED_DIR, "split_train_idx.npy"))
    test_idx = np.load(os.path.join(EMBED_DIR, "split_test_idx.npy"))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train_bin, y_test_bin = y_bin[train_idx], y_bin[test_idx]
    y_train_pol, y_test_pol = y_pol[train_idx], y_pol[test_idx]

    # L2 normalization (same as IQC)
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test  /= np.linalg.norm(X_test, axis=1, keepdims=True)

    return X_train, X_test, y_train_bin, y_test_bin, y_train_pol, y_test_pol


def eval_classical_models(X_train, X_test, y_train_bin, y_test_bin):
    results = []

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            solver="lbfgs"
        ),
        "Linear SVM": SVC(
            kernel="linear"
        ),
        "RBF SVM": SVC(
            kernel="rbf",
            gamma="scale"
        ),
        "k-NN": KNeighborsClassifier(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train_bin)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test_bin, y_pred)
        results.append((name, acc))

    return results


def eval_adaptive_iqc(X_train, X_test, y_train_pol, y_test_pol):
    backend = ExactBackend()

    # Bootstrap 1 memory per class
    class_states = []
    for cls in [-1, +1]:
        idx = np.where(y_train_pol == cls)[0][0]
        chi = X_train[idx].astype(np.complex128)
        chi /= np.linalg.norm(chi)
        class_states.append(
            ClassState(chi, label=cls, backend=backend)
        )

    memory_bank = MemoryBank(class_states)

    learner = Regime4ASpawn(
        memory_bank=memory_bank,
        eta=0.1,
        backend=backend,
        delta_cover=0.2,
        spawn_cooldown=100,
        min_polarized_per_class=1,
    )

    pruner = Regime4BPruning(
        memory_bank=memory_bank,
        tau_harm=-0.15,
        min_age=200,
        min_per_class=1,
        prune_interval=200,
    )

    model = AdaptiveMemoryModel(
        memory_bank=memory_bank,
        learner=learner,
        pruner=pruner,
        tau_responsible=0.1,
        beta=0.98,
    )

    # Adaptive training
    model.fit(X_train, y_train_pol)

    # Consolidation
    model.consolidate(
        X_train,
        y_train_pol,
        epochs=5,
        eta_scale=0.3,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_pol, y_pred)
    mem = len(memory_bank.class_states)

    return acc, mem


def main():
    print("\n📊 Adaptive IQC vs Classical Models\n")

    Xtr, Xte, ytr_bin, yte_bin, ytr_pol, yte_pol = load_data()

    # Classical models
    classical_results = eval_classical_models(
        Xtr, Xte, ytr_bin, yte_bin
    )

    # Adaptive IQC
    iqc_acc, iqc_mem = eval_adaptive_iqc(
        Xtr, Xte, ytr_pol, yte_pol
    )

    print("\n=== Classical Models ===")
    for name, acc in classical_results:
        print(f"{name:25s} | Acc: {acc:.4f}")

    print("\n=== Adaptive Quantum Model ===")
    print(
        f"Adaptive IQC (consolidated) | "
        f"Acc: {iqc_acc:.4f} | Memory: {iqc_mem}"
    )


if __name__ == "__main__":
    main()
