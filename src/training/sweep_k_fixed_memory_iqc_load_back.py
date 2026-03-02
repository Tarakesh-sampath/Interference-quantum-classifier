import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

from src.utils.load_data import load_data
from src.IQL.models.fixed_memory_iqc import FixedMemoryIQC
from src.IQL.backends.exact import ExactBackend
from src.IQL.backends.hardware_native import HardwareNativeBackend
from src.utils.paths import load_paths
from src.utils.seed import set_seed

def main():
    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    USE_EXACT_BACKEND = True  # Set to False to use the saved backend (HardwareNativeBackend)
    
    # -------------------------------------------------
    # Load paths and Set Seed
    # -------------------------------------------------
    set_seed()
    BASE_ROOT, PATHS = load_paths()

    OUT_DIR = os.path.join(BASE_ROOT, "results", "fixed_memory_iqc_sweep")
    MODELS_DIR = os.path.join(OUT_DIR, "models")
    
    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = load_data("polar")

    # Quantum-safe normalization (defensive)
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    # Find all saved models
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Models directory not found: {MODELS_DIR}")
        return

    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")])
    if not model_files:
        print(f"❌ No models found in {MODELS_DIR}")
        return

    k_values = []
    accuracies = []

    print(f"🚀 Starting evaluation of {len(model_files)} models...")
    if USE_EXACT_BACKEND:
        print("⚡ Using ExactBackend for evaluation.")
    else:
        print("🔌 Using saved backends (HardwareNativeBackend).")

    for model_name in model_files:
        # Extract K from filename (e.g., fixed_memory_iqc_k10.pkl)
        try:
            k_val = int(model_name.split("_k")[1].split(".")[0])
        except (IndexError, ValueError):
            print(f"⚠️ Warning: Could not parse K value from {model_name}, skipping.")
            continue

        model_path = os.path.join(MODELS_DIR, model_name)
        
        # Load model
        model = FixedMemoryIQC.load(model_path)
        
        # Swap backend if requested
        if USE_EXACT_BACKEND:
            exact_backend = ExactBackend()
            model.backend = exact_backend
            # Crucially, update the individual class states
            if model.memory_bank:
                for cs in model.memory_bank.class_states:
                    cs.backend = exact_backend
        
        start_time = time.time()
        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        eval_time = time.time() - start_time
        print(f"✅ Loaded K={k_val} | Test Accuracy: {acc:.4f} | Eval Time: {eval_time:.2f}s")
        
        k_values.append(k_val)
        accuracies.append(acc)

    # Sort results by K
    results = sorted(zip(k_values, accuracies))
    k_values, accuracies = zip(*results)

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    suffix = "_exact" if USE_EXACT_BACKEND else ""
    plot_path = os.path.join(OUT_DIR, f"k_vs_accuracy_eval{suffix}.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='g', label='Evaluation Accuracy')
    plt.title(f"FixedMemoryIQC Evaluation: K vs Accuracy {'(Exact Backend)' if USE_EXACT_BACKEND else ''}")
    plt.xlabel("K (Memory Size)")
    plt.ylabel("Test Accuracy")
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 K vs Accuracy plot saved to: {plot_path}")

    # Save results summary
    summary_path = os.path.join(OUT_DIR, f"eval_results{suffix}.txt")
    with open(summary_path, "w") as f:
        f.write("K\tAccuracy\n")
        for k, acc in zip(k_values, accuracies):
            f.write(f"{k}\t{acc:.4f}\n")
    print(f"📝 Evaluation results summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
