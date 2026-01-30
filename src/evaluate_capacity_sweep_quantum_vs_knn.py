import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar, ensure_binary

# Quantum models
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
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

    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test  /= np.linalg.norm(X_test, axis=1, keepdims=True)

    return X_train, X_test, y_train_bin, y_test_bin, y_train_pol, y_test_pol


def eval_fixed_iqc_k_sweep(Xtr, Xte, ytr_pol, yte_pol, K_values):
    accs = []
    for K in K_values:
        model = FixedMemoryIQC(K=K, eta=0.1)
        model.fit(Xtr, ytr_pol)
        y_pred = model.predict(Xte)
        acc = accuracy_score(yte_pol, y_pred)
        accs.append(acc)
        print(f"Fixed IQC | K={K:<2} | Acc={acc:.4f}")
    return accs


def eval_adaptive_initial_k_sweep(Xtr, Xte, ytr_pol, yte_pol, K_values):
    accs, final_sizes = [], []

    for K in K_values:
        backend = ExactBackend()
        class_states = []

        for cls in [-1, +1]:
            idxs = np.where(ytr_pol == cls)[0][:K]
            for idx in idxs:
                chi = Xtr[idx].astype(np.complex128)
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

        model.fit(Xtr, ytr_pol)
        model.consolidate(Xtr, ytr_pol, epochs=5, eta_scale=0.3)

        y_pred = model.predict(Xte)
        acc = accuracy_score(yte_pol, y_pred)

        accs.append(acc)
        final_sizes.append(len(memory_bank.class_states))

        print(
            f"Adaptive IQC | init K={K:<2} | "
            f"final mem={final_sizes[-1]:<2} | Acc={acc:.4f}"
        )

    return accs, final_sizes


def eval_knn_k_sweep(Xtr, Xte, ytr_bin, yte_bin, k_values):
    accs = []
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(Xtr, ytr_bin)
        y_pred = clf.predict(Xte)
        acc = accuracy_score(yte_bin, y_pred)
        accs.append(acc)
        print(f"k-NN | k={k:<2} | Acc={acc:.4f}")
    return accs


def main():
    Xtr, Xte, ytr_bin, yte_bin, ytr_pol, yte_pol = load_data()

    K = [i for i in range(1,20)]

    print("\n=== FixedMemory IQC sweep ===")
    fixed_acc = eval_fixed_iqc_k_sweep(Xtr, Xte, ytr_pol, yte_pol, K)

    print("\n=== Adaptive IQC sweep ===")
    adapt_acc, adapt_sizes = eval_adaptive_initial_k_sweep(
        Xtr, Xte, ytr_pol, yte_pol, K
    )

    print("\n=== k-NN sweep ===")
    knn_acc = eval_knn_k_sweep(Xtr, Xte, ytr_bin, yte_bin, K)

    # ------------------ Plot ------------------
    plt.figure(figsize=(7, 5))
    plt.plot(K, fixed_acc, marker="o", label="FixedMemory IQC")
    plt.plot(K, adapt_acc, marker="s", label="Adaptive IQC")
    plt.plot(K, knn_acc, marker="^", label="k-NN")

    plt.xlabel("Capacity parameter (K or k)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Capacity: Quantum IQC vs k-NN")
    plt.legend()
    plt.grid(True)

    _, PATHS = load_paths()
    out = os.path.join(PATHS["figures"], "capacity_sweep_quantum_vs_knn.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    print(f"\n📈 Plot saved to: {out}")


if __name__ == "__main__":
    main()
