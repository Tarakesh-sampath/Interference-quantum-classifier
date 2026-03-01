# src/training/protocol_adaptive/test_adaptive_memory_trainer.py

import os
import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.paths import load_paths
from src.utils.load_data import load_data

from src.IQL.learning.class_state import ClassState
from src.IQL.learning.memory_bank import MemoryBank

from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend
from src.IQL.regimes.regime4a_spawn import Regime4ASpawn
from src.IQL.regimes.regime4b_pruning import Regime4BPruning
from src.IQL.models.adaptive_memory_model import AdaptiveMemoryModel

import matplotlib.pyplot as plt
from collections import Counter


def main():
    print("\n🚀 Testing AdaptiveMemoryTrainer (V0)\n")

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # -------------------------------------------------
    # Bootstrap initial memory (1 per class)
    # -------------------------------------------------
    backend = HardwareNativeBackend()
    class_states = []

    for cls in [-1, +1]:
        idx = np.where(y_train == cls)[0][0]
        chi = X_train[idx].astype(np.complex128)
        chi /= np.linalg.norm(chi)
        class_states.append(
            ClassState(chi, label=cls, backend=backend)
        )

    memory_bank = MemoryBank(class_states)
    print("Initial memory size:", len(memory_bank.class_states))

    # -------------------------------------------------
    # Regime-4A (spawn)
    # -------------------------------------------------
    learner = Regime4ASpawn(
        memory_bank=memory_bank,
        eta=0.1,
        backend=backend,
        delta_cover=0.2,
        spawn_cooldown=100,
        min_polarized_per_class=1,
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
    # Adaptive trainer
    # -------------------------------------------------
    trainer = AdaptiveMemoryModel(
        memory_bank=memory_bank,
        learner=learner,
        pruner=pruner,
        tau_responsible=0.1,
        beta=0.98,
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    trainer.fit(X_train, y_train)

    # -------------------------------------------------
    # Consolidation phase
    # -------------------------------------------------
    trainer.consolidate(
        X_train,
        y_train,
        epochs=5,
        eta_scale=0.3,
    )

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------
    y_pred = trainer.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Adaptive Trainer Summary ===")
    print(trainer.summary())

    print("\n=== Evaluation ===")
    print(f"Test Accuracy      : {acc:.4f}")
    print(f"Final Memory Size  : {len(memory_bank.class_states)}")

    print("\n✅ AdaptiveMemoryTrainer test completed.\n")

    # -------------------------------------------------
    # Save adaptive diagnostics
    # -------------------------------------------------
    RESULTS_DIR = "results/figures/adaptive"
    save_adaptive_plots(trainer, memory_bank, RESULTS_DIR)
    #memory_bank.visualize(
    #    qubit=0,
    #    title="Adaptive IQC – Memory States (Final Snapshot)",
    #    save_path=os.path.join(RESULTS_DIR, "memory_states.png"),
    #    show=True,
    #)

def save_adaptive_plots(trainer, memory_bank, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # 1. Memory size over time
    # -------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(trainer.history["memory_size"])
    plt.xlabel("Training step")
    plt.ylabel("Memory size")
    plt.title("Adaptive Memory Size Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "memory_size_over_time.png"))
    plt.close()

    # -------------------------------------------------
    # 2. Action distribution
    # -------------------------------------------------
    action_counts = Counter(trainer.history["action"])

    plt.figure(figsize=(5, 4))
    plt.bar(action_counts.keys(), action_counts.values())
    plt.xlabel("Action type")
    plt.ylabel("Count")
    plt.title("Adaptive Actions Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_distribution.png"))
    plt.close()

    # -------------------------------------------------
    # 3. Harm EMA distribution
    # -------------------------------------------------
    harm = [cs.harm_ema for cs in memory_bank.class_states]

    plt.figure(figsize=(6, 4))
    plt.hist(harm, bins=20)
    plt.axvline(x=0.0, linestyle="--")
    plt.xlabel("Harm EMA")
    plt.ylabel("Count")
    plt.title("Harm EMA Distribution (Final)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "harm_ema_distribution.png"))
    plt.close()

    # -------------------------------------------------
    # 4. Memory age distribution
    # -------------------------------------------------
    ages = [cs.age for cs in memory_bank.class_states]

    plt.figure(figsize=(6, 4))
    plt.hist(ages, bins=15)
    plt.xlabel("Memory age")
    plt.ylabel("Count")
    plt.title("Memory Age Distribution (Final)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "memory_age_distribution.png"))
    plt.close()

    print(f"\n📊 Adaptive plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
