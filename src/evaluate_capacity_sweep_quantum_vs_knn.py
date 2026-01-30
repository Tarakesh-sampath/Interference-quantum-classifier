import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar, ensure_binary
from src.utils.load_data import load_data

# Quantum models
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.models.adaptive_memory_model import AdaptiveMemoryModel
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning



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
    Xtr, Xte, ytr_bin, yte_bin, ytr_pol, yte_pol = load_data("all")

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


## output

"""

=== FixedMemory IQC sweep ===
🌱 Global seed set to 42
Fixed IQC | K=1  | Acc=0.9000
🌱 Global seed set to 42
Fixed IQC | K=2  | Acc=0.8927
🌱 Global seed set to 42
Fixed IQC | K=3  | Acc=0.8827
🌱 Global seed set to 42
Fixed IQC | K=4  | Acc=0.8947
🌱 Global seed set to 42
Fixed IQC | K=5  | Acc=0.8927
🌱 Global seed set to 42
Fixed IQC | K=6  | Acc=0.8913
🌱 Global seed set to 42
Fixed IQC | K=7  | Acc=0.8900
🌱 Global seed set to 42
Fixed IQC | K=8  | Acc=0.8847
🌱 Global seed set to 42
Fixed IQC | K=9  | Acc=0.8873
🌱 Global seed set to 42
Fixed IQC | K=10 | Acc=0.8893
🌱 Global seed set to 42
Fixed IQC | K=11 | Acc=0.8913
🌱 Global seed set to 42
Fixed IQC | K=12 | Acc=0.8807
🌱 Global seed set to 42
Fixed IQC | K=13 | Acc=0.8867
🌱 Global seed set to 42
Fixed IQC | K=14 | Acc=0.8893
🌱 Global seed set to 42
Fixed IQC | K=15 | Acc=0.8920
🌱 Global seed set to 42
Fixed IQC | K=16 | Acc=0.8860
🌱 Global seed set to 42
Fixed IQC | K=17 | Acc=0.8887
🌱 Global seed set to 42
Fixed IQC | K=18 | Acc=0.8840
🌱 Global seed set to 42
Fixed IQC | K=19 | Acc=0.8853

=== Adaptive IQC sweep ===

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=1  | final mem=6  | Acc=0.8967

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=2  | final mem=4  | Acc=0.8773

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=3  | final mem=4  | Acc=0.8787

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=4  | final mem=8  | Acc=0.8660

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=5  | final mem=8  | Acc=0.8980

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=6  | final mem=7  | Acc=0.8847

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=7  | final mem=10 | Acc=0.8827

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=8  | final mem=11 | Acc=0.8620

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=9  | final mem=17 | Acc=0.8687

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=10 | final mem=21 | Acc=0.8833

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=11 | final mem=21 | Acc=0.8780

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=12 | final mem=21 | Acc=0.8860

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=13 | final mem=21 | Acc=0.8853

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=14 | final mem=23 | Acc=0.8653

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=15 | final mem=24 | Acc=0.8800

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=16 | final mem=28 | Acc=0.8727

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=17 | final mem=20 | Acc=0.8500

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=18 | final mem=26 | Acc=0.8660

🔒 Consolidation phase started (epochs=5, eta_scale=0.3)
  ✔ Consolidation epoch 1/5
  ✔ Consolidation epoch 2/5
  ✔ Consolidation epoch 3/5
  ✔ Consolidation epoch 4/5
  ✔ Consolidation epoch 5/5
🔓 Consolidation phase completed

Adaptive IQC | init K=19 | final mem=23 | Acc=0.8820

=== k-NN sweep ===
k-NN | k=1  | Acc=0.9140
k-NN | k=2  | Acc=0.9187
k-NN | k=3  | Acc=0.9233
k-NN | k=4  | Acc=0.9340
k-NN | k=5  | Acc=0.9260
k-NN | k=6  | Acc=0.9300
k-NN | k=7  | Acc=0.9267
k-NN | k=8  | Acc=0.9307
k-NN | k=9  | Acc=0.9260
k-NN | k=10 | Acc=0.9267
k-NN | k=11 | Acc=0.9267
k-NN | k=12 | Acc=0.9253
k-NN | k=13 | Acc=0.9247
k-NN | k=14 | Acc=0.9253
k-NN | k=15 | Acc=0.9247
k-NN | k=16 | Acc=0.9227
k-NN | k=17 | Acc=0.9233
k-NN | k=18 | Acc=0.9233
k-NN | k=19 | Acc=0.9220
"""