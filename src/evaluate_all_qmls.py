import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.label_utils import ensure_polar, ensure_binary
from src.utils.load_data import load_data
# Models
from src.IQL.models.static_isdo_model import StaticISDOModel
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.models.adaptive_memory_model import AdaptiveMemoryModel

# Adaptive components
from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank
from src.IQL.backends.exact import ExactBackend
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning


def eval_static_isdo(X_train, X_test, y_train_bin, y_test_bin):
    model = StaticISDOModel(K=3)
    model.fit(X_train, y_train_bin)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_bin, y_pred)
    return acc, "static", 6  # 2*K memories


def eval_fixed_iqc(X_train, X_test, y_train_pol, y_test_pol):
    model = FixedMemoryIQC(K=3, eta=0.1)
    model.fit(X_train, y_train_pol)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_pol, y_pred)
    mem = len(model.memory_bank.class_states)
    return acc, "fixed", mem


def eval_adaptive_iqc(X_train, X_test, y_train_pol, y_test_pol):
    backend = ExactBackend()

    # Bootstrap memory (1 per class)
    class_states = []
    for cls in [-1, +1]:
        idx = np.where(y_train_pol == cls)[0][0]
        chi = X_train[idx].astype(np.complex128)
        chi /= np.linalg.norm(chi)
        class_states.append(ClassState(chi, label=cls, backend=backend))

    memory_bank = MemoryBank(class_states)

    learner = Regime4ASpawn(
        memory_bank=memory_bank,
        eta=0.1,
        backend=backend,
        delta_cover=0.2,
        spawn_cooldown=100,
        min_polarized_per_class=2,
    )

    pruner = Regime4BPruning(
        memory_bank=memory_bank,
        tau_harm=-0.15,
        min_age=100,
        min_per_class=1,
        prune_interval=150,
    )

    model = AdaptiveMemoryModel(
        memory_bank=memory_bank,
        learner=learner,
        pruner=pruner,
        tau_responsible=0.1,
        beta=0.98,
    )

    # Adaptive phase
    model.fit(X_train, y_train_pol)

    # Consolidation phase
    model.consolidate(X_train, y_train_pol, epochs=5, eta_scale=0.4)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_pol, y_pred)
    mem = len(memory_bank.class_states)

    return acc, "adaptive", mem


def main():
    print("\n📊 Unified Model Evaluation\n")

    Xtr, Xte, ytr_bin, yte_bin, ytr_pol, yte_pol = load_data("all")

    results = []

    acc, typ, mem = eval_static_isdo(Xtr, Xte, ytr_bin, yte_bin)
    results.append(("Static ISDO", acc, typ, mem))

    acc, typ, mem = eval_fixed_iqc(Xtr, Xte, ytr_pol, yte_pol)
    results.append(("FixedMemory IQC (K=3)", acc, typ, mem))

    acc, typ, mem = eval_adaptive_iqc(Xtr, Xte, ytr_pol, yte_pol)
    results.append(("Adaptive IQC (with consolidation)", acc, typ, mem))

    print("\n=== Final Comparison ===")
    for name, acc, typ, mem in results:
        print(f"{name:35s} | Acc: {acc:.4f} | Type: {typ:8s} | Memory: {mem}")


if __name__ == "__main__":
    main()

"""

📊 Unified Model Evaluation

No frames directory specified. Skipping frame saving.                                                                              

=== Final Comparison ===
Static ISDO                         | Acc: 0.8807 | Type: static   | Memory: 6
FixedMemory IQC (K=3)               | Acc: 0.8827 | Type: fixed    | Memory: 6
Adaptive IQC (with consolidation)   | Acc: 0.9000 | Type: adaptive | Memory: 3
"""